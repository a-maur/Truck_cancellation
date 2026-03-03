#!/usr/bin/env python3
"""IQN optimiser for truck-cancellation decisions.

This optimiser treats each row as a one-step transition:
- state: route/day/hour features
- action: 0=keep last truck, 1=cancel last truck
- reward: configurable reward matrix

Training uses:
- IQN critic with quantile Huber loss
- replay buffer (uniform or prioritized)
- Double-Q target selection with target network
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import optimizers

try:
    from .base import (
        DataBundle,
        FeatureConfig,
        IQNQNetwork,
        LabelConfig,
        OneStepCancellationEnv,
        PrioritizedReplayBuffer,
        RewardConfig,
        apply_gradients_clipped,
        build_replay_buffer,
        choose_actions,
        compute_decision_metrics,
        compute_rewards,
        default_dataset_paths,
        hard_update_model,
        linear_schedule,
        load_data_bundle,
        quantile_huber_loss_per_sample,
        reduce_per_sample_loss,
        sample_replay_batch,
        save_bundle_metadata,
        save_json,
        seed_everything,
        sigmoid_np,
        soft_update_model,
    )
    from .config_ppo import (
        default_config_path,
        list_stage_trials,
        load_config as load_iqn_config,
        resolve_run_defaults,
    )
except ImportError:
    from base import (
        DataBundle,
        FeatureConfig,
        IQNQNetwork,
        LabelConfig,
        OneStepCancellationEnv,
        PrioritizedReplayBuffer,
        RewardConfig,
        apply_gradients_clipped,
        build_replay_buffer,
        choose_actions,
        compute_decision_metrics,
        compute_rewards,
        default_dataset_paths,
        hard_update_model,
        linear_schedule,
        load_data_bundle,
        quantile_huber_loss_per_sample,
        reduce_per_sample_loss,
        sample_replay_batch,
        save_bundle_metadata,
        save_json,
        seed_everything,
        sigmoid_np,
        soft_update_model,
    )
    from config_ppo import (
        default_config_path,
        list_stage_trials,
        load_config as load_iqn_config,
        resolve_run_defaults,
    )


def default_output_root_path() -> Path:
    return Path(__file__).resolve().parent / "outputs"


@dataclass
class IQNConfig:
    """Hyperparameters for one-step IQN training."""

    seed: int = 42
    updates: int = 300
    rollout_size: int = 4096
    batch_size: int = 1024
    grad_steps_per_update: int = 4
    min_replay_size: int = 4096
    replay_capacity: int = 200000

    gamma: float = 1.0
    lr: float = 3e-4
    grad_clip_norm: float = 10.0
    hidden_sizes: tuple[int, ...] = (128, 128)

    n_quantiles: int = 64
    n_target_quantiles: int = 64
    n_cos: int = 64
    kappa: float = 1.0
    dueling: bool = True
    noisy_nets: bool = False

    target_update_mode: str = "polyak"  # polyak|hard
    target_update_every: int = 10
    target_update_tau: float = 0.01

    use_per: bool = True
    per_alpha: float = 0.6
    per_beta0: float = 0.4
    per_beta1: float = 1.0
    per_eps: float = 1e-6

    epsilon_start: float = 0.20
    epsilon_end: float = 0.02
    epsilon_decay_updates: int = 200
    decision_temperature: float = 1.0

    eval_every: int = 10


def _format_minutes_seconds(seconds: float) -> str:
    total_seconds = max(0, int(round(float(seconds))))
    minutes, rem = divmod(total_seconds, 60)
    return f"{minutes}m {rem:02d}s"


def _build_hour_axis(*hour_arrays: np.ndarray | None) -> np.ndarray:
    mins: list[int] = []
    maxs: list[int] = []
    for arr in hour_arrays:
        if arr is None:
            continue
        h = np.asarray(arr, dtype=np.float32).reshape(-1)
        if h.size == 0:
            continue
        mins.append(int(np.floor(np.min(h))))
        maxs.append(int(np.ceil(np.max(h))))
    if not mins:
        return np.arange(0, 1, dtype=np.int32)
    return np.arange(min(mins), max(maxs) + 1, dtype=np.int32)


def _hourly_count(hour_axis: np.ndarray, hours: np.ndarray | None) -> np.ndarray:
    out = np.zeros((hour_axis.shape[0],), dtype=np.float32)
    if hours is None:
        return out
    h = np.asarray(hours, dtype=np.int32).reshape(-1)
    for i, hour in enumerate(hour_axis):
        out[i] = float(np.sum(h == int(hour)))
    return out


def _extract_corr_label(plot_context: dict | None) -> str | None:
    if not isinstance(plot_context, dict):
        return None

    candidates: list[tuple[object, object]] = [
        (plot_context.get("sim_correlation_dest"), plot_context.get("sim_correlation_type")),
        (plot_context.get("dataset_sim_correlation_dest"), plot_context.get("dataset_sim_correlation_type")),
    ]
    dataset_payload = plot_context.get("dataset")
    if isinstance(dataset_payload, dict):
        candidates.append(
            (
                dataset_payload.get("sim_correlation_dest"),
                dataset_payload.get("sim_correlation_type"),
            )
        )

    for corr_dest_raw, corr_type_raw in candidates:
        try:
            corr_dest = float(corr_dest_raw) if corr_dest_raw is not None else None
            corr_type = float(corr_type_raw) if corr_type_raw is not None else None
        except Exception:
            continue
        if corr_dest is None or corr_type is None:
            continue
        return f"corr = ({corr_dest:.3g}, {corr_type:.3g})"
    return None


def _load_dataset_sim_context(train_path: str | Path) -> dict[str, float]:
    dataset_info_path = Path(train_path).expanduser().resolve().parent / "dataset_info.json"
    if not dataset_info_path.exists():
        return {}
    try:
        with dataset_info_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    sim = payload.get("simulation")
    if not isinstance(sim, dict):
        return {}

    out: dict[str, float] = {}
    corr_dest = sim.get("correlation_dest")
    corr_type = sim.get("correlation_type")
    try:
        if corr_dest is not None:
            out["sim_correlation_dest"] = float(corr_dest)
    except Exception:
        pass
    try:
        if corr_type is not None:
            out["sim_correlation_type"] = float(corr_type)
    except Exception:
        pass
    return out


class IQNTruckCancellationOptimiser:
    """Offline IQN trainer for binary cancel/keep decisions."""

    def __init__(
        self,
        data: DataBundle,
        reward_config: RewardConfig | None = None,
        cfg: IQNConfig | None = None,
    ):
        self.data = data
        self.reward_cfg = reward_config or RewardConfig()
        self.cfg = cfg or IQNConfig()
        self.rng = seed_everything(self.cfg.seed)
        self.num_actions = 2

        self.env = OneStepCancellationEnv(
            states=self.data.x_train,
            labels_needed=self.data.y_train,
            reward_config=self.reward_cfg,
            hours=self.data.train_hours,
            min_hour=self.data.min_hour,
            max_hour=self.data.max_hour,
            rng=self.rng,
        )

        self.q_online = IQNQNetwork(
            input_dim=self.data.state_dim,
            num_actions=self.num_actions,
            hidden_sizes=self.cfg.hidden_sizes,
            n_cos=self.cfg.n_cos,
            dueling=self.cfg.dueling,
            noisy=self.cfg.noisy_nets,
        )
        self.q_target = IQNQNetwork(
            input_dim=self.data.state_dim,
            num_actions=self.num_actions,
            hidden_sizes=self.cfg.hidden_sizes,
            n_cos=self.cfg.n_cos,
            dueling=self.cfg.dueling,
            noisy=self.cfg.noisy_nets,
        )
        self.q_policy = IQNQNetwork(
            input_dim=self.data.state_dim,
            num_actions=self.num_actions,
            hidden_sizes=self.cfg.hidden_sizes,
            n_cos=self.cfg.n_cos,
            dueling=self.cfg.dueling,
            noisy=self.cfg.noisy_nets,
        )

        dummy_x = tf.zeros((1, self.data.state_dim), dtype=tf.float32)
        dummy_tau = tf.zeros((1, max(1, int(self.cfg.n_quantiles))), dtype=tf.float32)
        _ = self.q_online(dummy_x, dummy_tau, training=False)
        _ = self.q_target(dummy_x, dummy_tau, training=False)
        _ = self.q_policy(dummy_x, dummy_tau, training=False)
        hard_update_model(self.q_online, self.q_target)
        hard_update_model(self.q_online, self.q_policy)

        self.optimizer = optimizers.Adam(learning_rate=float(self.cfg.lr))
        self.replay = build_replay_buffer(
            obs_dim=self.data.state_dim,
            capacity=int(self.cfg.replay_capacity),
            use_per=bool(self.cfg.use_per),
            per_alpha=float(self.cfg.per_alpha),
            per_eps=float(self.cfg.per_eps),
        )

        self.history: list[dict[str, float]] = []
        self.training_status: dict[str, object] = {
            "requested_updates": int(self.cfg.updates),
            "executed_updates": 0,
            "stop_reason": "max_updates_reached",
            "replay_type": "prioritized" if isinstance(self.replay, PrioritizedReplayBuffer) else "uniform",
            "use_per": bool(self.cfg.use_per),
            "target_update_mode": str(self.cfg.target_update_mode),
            "target_update_every": int(self.cfg.target_update_every),
            "target_update_tau": float(self.cfg.target_update_tau),
        }
        self._grad_steps = 0

    def _sample_taus(self, batch_size: int, n_quantiles: int) -> tf.Tensor:
        b = int(max(1, batch_size))
        n = int(max(1, n_quantiles))
        return tf.random.uniform((b, n), minval=0.0, maxval=1.0, dtype=tf.float32)

    def _per_beta(self, update: int) -> float:
        t = min(max(0, int(update) - 1), max(1, int(self.cfg.updates)) - 1)
        return linear_schedule(
            float(self.cfg.per_beta0),
            float(self.cfg.per_beta1),
            step=t,
            total_steps=max(2, int(self.cfg.updates)),
        )

    def _sync_policy(self) -> None:
        hard_update_model(self.q_online, self.q_policy)

    def _maybe_update_target(self) -> None:
        mode = str(self.cfg.target_update_mode).strip().lower()
        if mode == "polyak":
            soft_update_model(self.q_online, self.q_target, tau=float(self.cfg.target_update_tau))
            return
        if mode == "hard":
            if self._grad_steps % max(1, int(self.cfg.target_update_every)) == 0:
                hard_update_model(self.q_online, self.q_target)
            return
        raise ValueError(
            f"Unknown target_update_mode={self.cfg.target_update_mode!r}. Expected polyak|hard."
        )

    def _mean_q_values(
        self,
        model: IQNQNetwork,
        states: np.ndarray,
        n_quantiles: int | None = None,
        batch_size: int = 8192,
    ) -> np.ndarray:
        x = np.asarray(states, dtype=np.float32)
        n = x.shape[0]
        out = np.zeros((n, self.num_actions), dtype=np.float32)
        nq = int(n_quantiles if n_quantiles is not None else self.cfg.n_quantiles)
        b = int(max(1, batch_size))
        for start in range(0, n, b):
            stop = min(n, start + b)
            obs = tf.convert_to_tensor(x[start:stop], dtype=tf.float32)
            taus = self._sample_taus(stop - start, nq)
            q_atoms = model(obs, taus, training=False)
            out[start:stop] = tf.reduce_mean(q_atoms, axis=-1).numpy().astype(np.float32)
        return out

    def _cancel_prob_from_q(self, q_mean: np.ndarray) -> np.ndarray:
        q = np.asarray(q_mean, dtype=np.float32)
        if q.ndim != 2 or q.shape[1] != self.num_actions:
            raise ValueError(f"Expected q_mean shape [B,{self.num_actions}], got {q.shape}")
        temp = max(1e-6, float(self.cfg.decision_temperature))
        logits = (q[:, 1] - q[:, 0]) / temp
        return sigmoid_np(logits)

    def _epsilon_greedy_actions(self, states: np.ndarray, epsilon: float) -> tuple[np.ndarray, np.ndarray]:
        q_mean = self._mean_q_values(self.q_online, states)
        greedy = np.argmax(q_mean, axis=1).astype(np.int32)
        eps = float(np.clip(epsilon, 0.0, 1.0))
        if eps <= 0.0:
            return greedy, q_mean
        actions = greedy.copy()
        explore = self.rng.random(actions.shape[0]) < eps
        if np.any(explore):
            actions[explore] = self.rng.integers(0, self.num_actions, size=int(np.sum(explore)), dtype=np.int32)
        return actions.astype(np.int32), q_mean

    def _collect_replay_batch(self, batch_size: int, epsilon: float) -> dict[str, float]:
        idx = self.env.sample_indices(batch_size)
        obs = self.env.states[idx].astype(np.float32)
        labels = self.env.labels[idx].astype(np.float32)
        hours = self.env.hours[idx].astype(np.float32)

        actions, _q_mean = self._epsilon_greedy_actions(obs, epsilon=epsilon)
        rewards = compute_rewards(
            actions,
            labels,
            self.reward_cfg,
            hours=hours,
            min_hour=self.env.min_hour,
            max_hour=self.env.max_hour,
        ).astype(np.float32)

        # One-step task: every sampled transition is terminal.
        next_obs = np.zeros_like(obs, dtype=np.float32)
        dones = np.ones((obs.shape[0],), dtype=np.float32)
        discounts = (float(self.cfg.gamma) * (1.0 - dones)).astype(np.float32)

        self.replay.add_batch(
            obs_batch=obs,
            act_batch=actions.astype(np.int32),
            rew_batch=rewards,
            next_obs_batch=next_obs,
            done_batch=dones,
            disc_batch=discounts,
        )
        return {
            "rollout_reward_mean": float(np.mean(rewards)),
            "rollout_cancel_rate": float(np.mean(actions == 1)),
        }

    def _train_step(self, per_beta: float) -> tuple[float, float]:
        sample = sample_replay_batch(self.replay, batch_size=int(self.cfg.batch_size), per_beta=float(per_beta))
        obs = tf.convert_to_tensor(sample.obs, dtype=tf.float32)
        nxt = tf.convert_to_tensor(sample.next_obs, dtype=tf.float32)
        actions = tf.convert_to_tensor(sample.actions, dtype=tf.int32)
        rewards = tf.convert_to_tensor(sample.rewards, dtype=tf.float32)
        discounts = tf.convert_to_tensor(sample.discounts, dtype=tf.float32)
        is_weights = tf.convert_to_tensor(sample.is_weights, dtype=tf.float32)

        batch_size = int(sample.obs.shape[0])
        n_cur = int(max(1, self.cfg.n_quantiles))
        n_tgt = int(max(1, self.cfg.n_target_quantiles))
        tau_cur = self._sample_taus(batch_size, n_cur)
        tau_next = self._sample_taus(batch_size, n_tgt)
        tau_tgt = self._sample_taus(batch_size, n_tgt)

        q_next_online = self.q_online(nxt, tau_next, training=False)
        next_actions = tf.argmax(tf.reduce_mean(q_next_online, axis=-1), axis=-1, output_type=tf.int32)
        q_next_target = self.q_target(nxt, tau_tgt, training=False)
        idx_next = tf.stack([tf.range(batch_size, dtype=tf.int32), next_actions], axis=1)
        theta_next = tf.gather_nd(q_next_target, idx_next)
        target_atoms = tf.stop_gradient(rewards[:, None] + discounts[:, None] * theta_next)

        with tf.GradientTape() as tape:
            q_atoms = self.q_online(obs, tau_cur, training=True)
            idx = tf.stack([tf.range(batch_size, dtype=tf.int32), actions], axis=1)
            theta_sa = tf.gather_nd(q_atoms, idx)
            td_errors = tf.expand_dims(target_atoms, axis=1) - tf.expand_dims(theta_sa, axis=2)
            loss_per_sample = quantile_huber_loss_per_sample(td_errors, tau_cur, kappa=float(self.cfg.kappa))
            loss = reduce_per_sample_loss(loss_per_sample, is_weights=is_weights)

        apply_gradients_clipped(
            optimizer=self.optimizer,
            loss=loss,
            variables=self.q_online.trainable_variables,
            tape=tape,
            grad_clip_norm=float(self.cfg.grad_clip_norm),
        )
        self._grad_steps += 1
        self._maybe_update_target()

        priorities = tf.reduce_mean(tf.abs(td_errors), axis=[1, 2]).numpy().astype(np.float32)
        if isinstance(self.replay, PrioritizedReplayBuffer) and sample.indices is not None:
            self.replay.update_priorities(sample.indices, priorities)

        td_abs_mean = float(np.mean(priorities)) if priorities.size else 0.0
        return float(loss.numpy()), td_abs_mean

    def _choose_eval_actions(self, q_mean: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray]:
        mode_norm = str(mode).strip().lower()
        cancel_prob = self._cancel_prob_from_q(q_mean)
        if mode_norm == "deterministic":
            actions = np.argmax(q_mean, axis=1).astype(np.int32)
            return actions, cancel_prob
        if mode_norm == "stochastic":
            actions = choose_actions(
                cancel_prob=cancel_prob,
                mode="stochastic",
                rng=self.rng,
                threshold=0.5,
                min_prob=0.0,
                max_prob=1.0,
            ).astype(np.int32)
            return actions, cancel_prob
        raise ValueError(f"Unknown mode={mode!r}. Expected deterministic|stochastic.")

    def evaluate_split(self, split: str = "test", mode: str = "deterministic") -> dict[str, float]:
        split_norm = str(split).strip().lower()
        if split_norm == "train":
            x = self.data.x_train
            y = self.data.y_train
            h = self.data.train_hours
        elif split_norm == "test":
            x = self.data.x_test
            y = self.data.y_test
            h = self.data.test_hours
        else:
            raise ValueError("split must be train|test")

        q_mean = self._mean_q_values(self.q_policy, x)
        actions, cancel_prob = self._choose_eval_actions(q_mean, mode=mode)
        rewards = compute_rewards(
            actions,
            y,
            self.reward_cfg,
            hours=h,
            min_hour=self.data.min_hour,
            max_hour=self.data.max_hour,
        )
        metrics = compute_decision_metrics(actions, y, rewards)
        metrics["avg_cancel_probability"] = float(np.mean(cancel_prob))
        return metrics

    def _predict_split_details(self, split: str = "test", mode: str = "deterministic") -> dict[str, np.ndarray]:
        split_norm = str(split).strip().lower()
        if split_norm == "train":
            x = self.data.x_train
            y = self.data.y_train
            h = self.data.train_hours
        elif split_norm == "test":
            x = self.data.x_test
            y = self.data.y_test
            h = self.data.test_hours
        else:
            raise ValueError("split must be train|test")

        q_mean = self._mean_q_values(self.q_policy, x)
        actions, cancel_prob = self._choose_eval_actions(q_mean, mode=mode)
        needed = (np.asarray(y, dtype=np.float32) >= 0.5).astype(np.int32)
        hours = np.asarray(h, dtype=np.int32) if h is not None else np.zeros((x.shape[0],), dtype=np.int32)
        return {
            "cancel_prob": np.asarray(cancel_prob, dtype=np.float32),
            "actions": np.asarray(actions, dtype=np.int32),
            "needed": needed,
            "hours": hours,
        }

    def _write_run_plots(
        self,
        out_dir: Path,
        final_metrics: dict[str, dict[str, float]] | None = None,
        plot_context: dict | None = None,
    ) -> list[str]:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:
            return []

        plot_dir = out_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        generated: list[str] = []

        def history_series(key: str) -> np.ndarray:
            vals: list[float] = []
            for row in self.history:
                raw = row.get(key)
                if raw is None:
                    vals.append(np.nan)
                else:
                    try:
                        vals.append(float(raw))
                    except Exception:
                        vals.append(np.nan)
            return np.asarray(vals, dtype=np.float32)

        def finite_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            mask = np.isfinite(np.asarray(y, dtype=np.float32))
            return np.asarray(x)[mask], np.asarray(y, dtype=np.float32)[mask]

        def cumulative_recall_from_hourly_tp(tp_hour: np.ndarray, n_positive: int) -> np.ndarray:
            """Monotonic cumulative recall: fraction of positives cancelled by each hour."""
            y = np.full(np.asarray(tp_hour).shape, np.nan, dtype=np.float32)
            if int(n_positive) <= 0:
                return y
            tp = np.asarray(tp_hour, dtype=np.float64)
            tp = np.where(np.isfinite(tp), tp, 0.0)
            cum = np.cumsum(tp)
            return np.clip(cum / float(n_positive), 0.0, 1.0).astype(np.float32)

        try:
            updates = np.asarray([int(row.get("update", 0)) for row in self.history], dtype=np.int32)
            if updates.size > 0:
                fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
                axes[0, 0].plot(updates, history_series("rollout_reward_mean"), linewidth=1.2)
                axes[0, 0].set_title("Rollout Reward Mean")
                axes[0, 0].set_ylabel("reward")
                axes[0, 0].grid(alpha=0.2)

                y_loss = history_series("q_loss")
                x_loss, y_loss = finite_xy(updates, y_loss)
                if y_loss.size > 0:
                    axes[0, 1].plot(x_loss, y_loss, linewidth=1.2, color="tab:red")
                axes[0, 1].set_title("IQN Loss")
                axes[0, 1].set_ylabel("loss")
                axes[0, 1].grid(alpha=0.2)

                y_acc = history_series("test_acc_det")
                x_acc, y_acc = finite_xy(updates, y_acc)
                if y_acc.size > 0:
                    axes[1, 0].plot(x_acc, y_acc, linewidth=1.2, color="tab:green", marker="o", markersize=3)
                axes[1, 0].set_title("Test Deterministic Accuracy")
                axes[1, 0].set_xlabel("update")
                axes[1, 0].set_ylabel("accuracy")
                axes[1, 0].set_ylim(-0.02, 1.02)
                axes[1, 0].grid(alpha=0.2)

                y_reward = history_series("test_reward_det")
                x_reward, y_reward = finite_xy(updates, y_reward)
                if y_reward.size > 0:
                    axes[1, 1].plot(
                        x_reward,
                        y_reward,
                        linewidth=1.2,
                        color="tab:orange",
                        marker="o",
                        markersize=3,
                    )
                axes[1, 1].set_title("Test Deterministic Reward")
                axes[1, 1].set_xlabel("update")
                axes[1, 1].set_ylabel("reward")
                axes[1, 1].grid(alpha=0.2)

                fig.tight_layout()
                path = plot_dir / "training_curves.png"
                fig.savefig(path, dpi=150)
                plt.close(fig)
                generated.append(path.name)
        except Exception:
            pass

        try:
            details = self._predict_split_details(split="test", mode="deterministic")
            test_actions = details["actions"].astype(np.int32)
            test_needed = details["needed"].astype(np.int32)
            test_hours = details["hours"].astype(np.int32)
            corr_label = _extract_corr_label(plot_context)

            if test_hours.size > 0:
                hour_axis = _build_hour_axis(test_hours)
                total = _hourly_count(hour_axis, test_hours)
                tp = np.zeros_like(total)
                fp = np.zeros_like(total)
                fn = np.zeros_like(total)
                for i, hour in enumerate(hour_axis):
                    m = test_hours == int(hour)
                    tp[i] = float(np.sum((test_actions[m] == 1) & (test_needed[m] == 0)))
                    fp[i] = float(np.sum((test_actions[m] == 1) & (test_needed[m] == 1)))
                    fn[i] = float(np.sum((test_actions[m] == 0) & (test_needed[m] == 0)))

                precision = tp / np.maximum(tp + fp, 1.0)
                positives_total = int(np.sum(test_needed == 0))
                recall = cumulative_recall_from_hourly_tp(tp_hour=tp, n_positive=positives_total)
                cancel_fraction = (tp + fp) / np.maximum(total, 1.0)

                fig, ax = plt.subplots(figsize=(10, 4.2))
                ax.plot(hour_axis, precision, color="tab:blue", label="PPV", linewidth=1.4)
                ax.plot(hour_axis, recall, color="tab:green", label="TPR (cumulative <= hour)", linewidth=1.4)
                ax.plot(hour_axis, cancel_fraction, color="darkorange", label="cancel fraction", linewidth=1.3)
                ax.set_xlabel("hour")
                ax.set_ylabel("fraction (correctly) cancelled")
                ax.set_ylim(-0.02, 1.02)
                title = "test hourly cancellation profile (cumulative TPR)"
                if corr_label:
                    title = f"{title} | {corr_label}"
                ax.set_title(title)
                ax.grid(alpha=0.25)
                ax.legend(loc="lower right")
                fig.tight_layout()
                path = plot_dir / "test_hourly_cancel_profile.png"
                fig.savefig(path, dpi=150)
                plt.close(fig)
                generated.append(path.name)

            # IQN diagnostic: inverse CDF + approximate PDF for best/worst action.
            x_test = np.asarray(self.data.x_test, dtype=np.float32)
            if x_test.ndim == 2 and x_test.shape[0] > 0:
                n_states = int(min(1024, x_test.shape[0]))
                sample_idx = self.rng.choice(x_test.shape[0], size=n_states, replace=(x_test.shape[0] < n_states))
                x_sel = x_test[sample_idx]
                n_tau_plot = int(np.clip(max(64, int(self.cfg.n_quantiles)), 64, 256))
                tau_grid = np.linspace(0.01, 0.99, n_tau_plot, dtype=np.float32)
                q_sum = np.zeros((self.num_actions, n_tau_plot), dtype=np.float64)
                n_batches = 0

                batch_plot = 256
                for start in range(0, x_sel.shape[0], batch_plot):
                    stop = min(x_sel.shape[0], start + batch_plot)
                    obs_tf = tf.convert_to_tensor(x_sel[start:stop], dtype=tf.float32)
                    tau_batch = np.broadcast_to(tau_grid[None, :], (stop - start, n_tau_plot)).astype(np.float32)
                    q_atoms = self.q_policy(
                        obs_tf,
                        tf.convert_to_tensor(tau_batch, dtype=tf.float32),
                        training=False,
                    ).numpy()
                    q_sum += np.mean(q_atoms, axis=0)
                    n_batches += 1

                if n_batches > 0:
                    q_inv_cdf = (q_sum / float(n_batches)).astype(np.float32)  # [A, N_tau]
                    mean_q = np.mean(q_inv_cdf, axis=1)  # [A]
                    best_action = int(np.argmax(mean_q))
                    worst_action = int(np.argmin(mean_q))
                    q_best = q_inv_cdf[best_action]
                    q_worst = q_inv_cdf[worst_action]

                    fig_dist, axes_dist = plt.subplots(2, 1, figsize=(8.6, 7.0))
                    ax_cdf, ax_pdf = axes_dist
                    ax_cdf.plot(
                        tau_grid,
                        q_best,
                        linewidth=1.35,
                        color="tab:blue",
                        label=f"inv CDF best action={best_action}",
                    )
                    ax_cdf.plot(
                        tau_grid,
                        q_worst,
                        linewidth=1.2,
                        color="tab:orange",
                        label=f"inv CDF worst action={worst_action}",
                    )
                    action_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
                    for action_idx in range(self.num_actions):
                        ax_cdf.axhline(
                            float(mean_q[action_idx]),
                            linestyle=":",
                            linewidth=1.0,
                            color=action_colors[action_idx % len(action_colors)],
                            alpha=0.9,
                            label=f"mean action={action_idx}",
                        )
                    ax_cdf.set_xlabel("tau (quantile level)")
                    ax_cdf.set_ylabel("Q^-1(tau)")
                    ax_cdf.grid(alpha=0.2)
                    ax_cdf.legend(loc="best", fontsize=8, frameon=False)

                    bins = max(20, int(np.sqrt(n_tau_plot)))
                    ax_pdf.hist(q_best, bins=bins, density=True, alpha=0.45, color="tab:blue", label="pdf(best)")
                    ax_pdf.hist(q_worst, bins=bins, density=True, alpha=0.45, color="tab:orange", label="pdf(worst)")
                    for action_idx in range(self.num_actions):
                        ax_pdf.axvline(
                            float(mean_q[action_idx]),
                            linestyle="--",
                            linewidth=1.0,
                            color=action_colors[action_idx % len(action_colors)],
                            alpha=0.9,
                            label=f"mean action={action_idx}",
                        )
                    ax_pdf.set_xlabel("return")
                    ax_pdf.set_ylabel("density")
                    ax_pdf.grid(alpha=0.2)
                    ax_pdf.legend(loc="best", fontsize=8, frameon=False)

                    dist_title = "IQN test return distribution: inverse CDF and approximate PDF"
                    if corr_label:
                        dist_title = f"{dist_title} | {corr_label}"
                    fig_dist.suptitle(dist_title, fontsize=11)
                    fig_dist.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
                    dist_path = plot_dir / "iqn_inv_cdf_pdf_test.pdf"
                    fig_dist.savefig(dist_path)
                    plt.close(fig_dist)
                    generated.append(dist_path.name)

            # Route-level hourly plots matching PPO output structure.
            test_df = self.data.test_df
            if (
                test_df is not None
                and hasattr(test_df, "columns")
                and all(col in test_df.columns for col in ("center", "dest", "hour"))
                and len(test_df) == test_actions.shape[0]
            ):
                center_values = np.asarray(test_df["center"].astype(str).to_numpy())
                dest_values = np.asarray(test_df["dest"].astype(str).to_numpy())
                hour_values = np.asarray(test_df["hour"].to_numpy(), dtype=np.int32)

                eval_df = pd.DataFrame(
                    {
                        "center": center_values,
                        "dest": dest_values,
                        "hour": hour_values,
                        "action": test_actions.astype(np.int32),
                        "needed": test_needed.astype(np.int32),
                    }
                )
                eval_df["cancelled"] = (eval_df["action"] == 1).astype(np.int32)
                eval_df["tp_cancel"] = (
                    (eval_df["action"] == 1) & (eval_df["needed"] == 0)
                ).astype(np.int32)
                eval_df["fp_cancel"] = (
                    (eval_df["action"] == 1) & (eval_df["needed"] == 1)
                ).astype(np.int32)
                eval_df["fn_cancel"] = (
                    (eval_df["action"] == 0) & (eval_df["needed"] == 0)
                ).astype(np.int32)

                agg = (
                    eval_df.groupby(["center", "dest", "hour"], as_index=False)
                    .agg(
                        total_trucks=("action", "size"),
                        cancellations=("cancelled", "sum"),
                        tp_cancel=("tp_cancel", "sum"),
                        fp_cancel=("fp_cancel", "sum"),
                        fn_cancel=("fn_cancel", "sum"),
                    )
                    .sort_values(["center", "dest", "hour"], kind="stable")
                    .reset_index(drop=True)
                )

                # Integrated hourly statistics across routes.
                agg_hour = agg.copy()
                denom_ppv_all = agg_hour["tp_cancel"] + agg_hour["fp_cancel"]
                agg_hour["ppv"] = np.where(
                    denom_ppv_all > 0,
                    agg_hour["tp_cancel"] / denom_ppv_all,
                    np.nan,
                )
                integrated = (
                    agg_hour.groupby("hour", as_index=False)
                    .agg(
                        ppv_mean=("ppv", "mean"),
                        tp_sum=("tp_cancel", "sum"),
                        cancellations=("cancellations", "sum"),
                    )
                    .sort_values("hour", kind="stable")
                    .reset_index(drop=True)
                )

                if not integrated.empty:
                    total_cancellations = float(np.sum(integrated["cancellations"].to_numpy(dtype=np.float64)))
                    cancel_share = np.zeros((integrated.shape[0],), dtype=np.float64)
                    if total_cancellations > 0.0:
                        cancel_share = integrated["cancellations"].to_numpy(dtype=np.float64) / total_cancellations

                    x_hour = integrated["hour"].to_numpy(dtype=np.float32)
                    y_ppv = integrated["ppv_mean"].to_numpy(dtype=np.float32)
                    positives_total = int(np.sum((agg["tp_cancel"] + agg["fn_cancel"]).to_numpy(dtype=np.int64)))
                    y_tpr = cumulative_recall_from_hourly_tp(
                        tp_hour=integrated["tp_sum"].to_numpy(dtype=np.float32),
                        n_positive=positives_total,
                    )

                    fig_int, ax_int = plt.subplots(figsize=(9.2, 4.8))
                    ax_int.plot(x_hour, y_ppv, marker="o", linewidth=1.35, linestyle="--", color="tab:blue", label="PPV")
                    ax_int.plot(
                        x_hour,
                        y_tpr,
                        marker="o",
                        linewidth=1.2,
                        linestyle="-",
                        color="darkorange",
                        label="TPR (cumulative <= hour)",
                    )
                    ax_int.plot(
                        x_hour,
                        cancel_share,
                        marker="o",
                        linewidth=1.2,
                        linestyle="-",
                        color="tab:red",
                        label="cancelled share by hour",
                    )
                    ax_int.set_xlabel("hour")
                    ax_int.set_ylabel("fraction")
                    ax_int.set_ylim(0.0, 1.0)
                    ax_int.grid(alpha=0.2)
                    ax_int.legend(loc="lower right", fontsize=9, frameon=False)

                    int_title = "Test: Integrated Hourly PPV, Cumulative TPR, and Cancellation Share"
                    if corr_label is not None:
                        int_title = f"{int_title} | {corr_label}"
                    ax_int.set_title(int_title, fontsize=12)
                    fig_int.tight_layout()
                    out_int_pdf = plot_dir / "integrated_statistics_per_hour.pdf"
                    fig_int.savefig(out_int_pdf)
                    plt.close(fig_int)
                    generated.append(out_int_pdf.name)

                centers_present = sorted(set(agg["center"].astype(str).tolist()))
                if centers_present:
                    all_dests_present = sorted(set(agg["dest"].astype(str).tolist()))
                    row_destinations: list[list[str]] = []
                    for c in centers_present:
                        default_dests = [d for d in all_dests_present if d != c]
                        center_dests = set(agg.loc[agg["center"].astype(str) == c, "dest"].astype(str).tolist())
                        row_destinations.append([d for d in default_dests if d in center_dests])

                    n_rows = len(centers_present)
                    n_cols = max((len(row) for row in row_destinations), default=0)
                    if n_cols > 0:
                        fig, axes = plt.subplots(
                            n_rows,
                            n_cols,
                            figsize=(3.0 * n_cols, 2.2 * n_rows),
                            sharex=True,
                            sharey=True,
                            squeeze=False,
                        )

                        for row_idx, center_key in enumerate(centers_present):
                            row_dests = row_destinations[row_idx]
                            for col_idx in range(n_cols):
                                ax = axes[row_idx][col_idx]
                                if col_idx >= len(row_dests):
                                    ax.set_axis_off()
                                    continue

                                dest_key = str(row_dests[col_idx])
                                lane = agg[
                                    (agg["center"].astype(str) == center_key)
                                    & (agg["dest"].astype(str) == dest_key)
                                ].sort_values("hour", kind="stable")
                                if lane.empty:
                                    ax.set_axis_off()
                                    continue

                                x_lane = lane["hour"].to_numpy(dtype=np.float32)
                                tp_lane = lane["tp_cancel"].to_numpy(dtype=np.float32)
                                fp_lane = lane["fp_cancel"].to_numpy(dtype=np.float32)
                                denom_ppv = tp_lane + fp_lane
                                y_lane_ppv = np.full_like(tp_lane, np.nan, dtype=np.float32)
                                np.divide(tp_lane, denom_ppv, out=y_lane_ppv, where=denom_ppv > 0.0)
                                ax.plot(
                                    x_lane,
                                    y_lane_ppv,
                                    marker="o",
                                    linewidth=1.1,
                                    linestyle="--",
                                    color="tab:blue",
                                    markersize=3.0,
                                )
                                ax.set_title(f"{center_key}->{dest_key}", fontsize=8)
                                ax.set_ylim(0.0, 1.0)
                                ax.grid(alpha=0.2)
                                if row_idx == n_rows - 1:
                                    ax.set_xlabel("hour", fontsize=8)
                                if col_idx == 0:
                                    ax.set_ylabel(f"{center_key}\nPPV", fontsize=8)
                                ax.tick_params(axis="both", labelsize=7)

                        fig.suptitle("Test: Hourly PPV by Route", fontsize=12)
                        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
                        out_grid = plot_dir / "hourly_correctly_cancelled_grid_test.png"
                        fig.savefig(out_grid, dpi=140)
                        plt.close(fig)
                        generated.append(out_grid.name)

                    for center_key in centers_present:
                        center_dests = sorted(
                            set(
                                agg.loc[agg["center"].astype(str) == center_key, "dest"]
                                .astype(str)
                                .tolist()
                            )
                        )
                        center_dests = [d for d in center_dests if d != str(center_key)]
                        if not center_dests:
                            continue

                        n_panels = len(center_dests)
                        n_cols_center = int(np.ceil(np.sqrt(n_panels)))
                        n_rows_center = int(np.ceil(float(n_panels) / float(max(1, n_cols_center))))
                        fig_center, axes_center = plt.subplots(
                            n_rows_center,
                            n_cols_center,
                            figsize=(3.4 * n_cols_center, 2.6 * n_rows_center),
                            sharex=True,
                            sharey=True,
                            squeeze=False,
                        )
                        flat_axes_center = axes_center.flatten()

                        for idx_dest, dest_key in enumerate(center_dests):
                            ax = flat_axes_center[idx_dest]
                            lane = agg[
                                (agg["center"].astype(str) == str(center_key))
                                & (agg["dest"].astype(str) == str(dest_key))
                            ].sort_values("hour", kind="stable")
                            if lane.empty:
                                ax.set_axis_off()
                                continue

                            x_hour = lane["hour"].to_numpy(dtype=np.float32)
                            tp_hour = lane["tp_cancel"].to_numpy(dtype=np.float32)
                            fp_hour = lane["fp_cancel"].to_numpy(dtype=np.float32)
                            fn_hour = lane["fn_cancel"].to_numpy(dtype=np.float32)
                            denom_ppv = tp_hour + fp_hour
                            y_ppv = np.full_like(tp_hour, np.nan, dtype=np.float32)
                            np.divide(tp_hour, denom_ppv, out=y_ppv, where=denom_ppv > 0.0)
                            positives_lane = int(np.sum((tp_hour + fn_hour).astype(np.int64)))
                            y_tpr = cumulative_recall_from_hourly_tp(
                                tp_hour=tp_hour,
                                n_positive=positives_lane,
                            )

                            ax.plot(
                                x_hour,
                                y_ppv,
                                marker="o",
                                linewidth=1.3,
                                linestyle="-",
                                color="tab:blue",
                                markersize=3.2,
                                label="PPV",
                            )
                            ax.plot(
                                x_hour,
                                y_tpr,
                                marker="o",
                                linewidth=1.15,
                                linestyle="-",
                                color="darkorange",
                                markersize=2.8,
                                label="TPR (cum)",
                            )
                            ax.set_title(f"{center_key} -> {dest_key}", fontsize=9)
                            ax.set_ylim(0.0, 1.0)
                            ax.grid(alpha=0.2)
                            ax.tick_params(axis="both", labelsize=8)
                            ax.legend(loc="lower right", fontsize=8, frameon=False)

                        for ax in flat_axes_center[n_panels:]:
                            ax.set_axis_off()

                        for r in range(n_rows_center):
                            for c in range(n_cols_center):
                                ax = axes_center[r][c]
                                if not ax.axison:
                                    continue
                                if r == n_rows_center - 1:
                                    ax.set_xlabel("hour", fontsize=9)
                                if c == 0:
                                    ax.set_ylabel("fraction", fontsize=9)

                        center_title = f"Test: PPV and Cumulative TPR per Hour ({center_key})"
                        if corr_label is not None:
                            center_title = f"{center_title} | {corr_label}"
                        fig_center.suptitle(center_title, fontsize=12)
                        fig_center.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
                        center_token = "".join(ch for ch in str(center_key) if ch.isalnum() or ch in {"_", "-"})
                        if not center_token:
                            center_token = "unknown"
                        out_pdf = plot_dir / f"center_{center_token}_dest_correctly_cancelled_trucks_per_hour.pdf"
                        fig_center.savefig(out_pdf)
                        plt.close(fig_center)
                        generated.append(out_pdf.name)
        except Exception:
            pass

        return generated

    def train(self) -> dict[str, dict[str, float]]:
        t0 = time.perf_counter()
        log_every_n_updates = 10
        executed_updates = 0

        for update in range(1, int(self.cfg.updates) + 1):
            log_this_update = (update % log_every_n_updates == 0) or (update == 1) or (update == self.cfg.updates)
            eps = linear_schedule(
                float(self.cfg.epsilon_start),
                float(self.cfg.epsilon_end),
                step=min(update - 1, max(1, int(self.cfg.epsilon_decay_updates)) - 1),
                total_steps=max(2, int(self.cfg.epsilon_decay_updates)),
            )
            beta = self._per_beta(update)

            t_collect = time.perf_counter()
            rollout_stats = self._collect_replay_batch(batch_size=int(self.cfg.rollout_size), epsilon=float(eps))
            collect_elapsed = time.perf_counter() - t_collect

            t_update = time.perf_counter()
            losses: list[float] = []
            td_abs_vals: list[float] = []
            if self.replay.can_sample(int(self.cfg.min_replay_size)):
                for _ in range(max(1, int(self.cfg.grad_steps_per_update))):
                    loss, td_abs = self._train_step(per_beta=beta)
                    losses.append(float(loss))
                    td_abs_vals.append(float(td_abs))
                self._sync_policy()
            update_elapsed = time.perf_counter() - t_update

            q_loss = float(np.mean(losses)) if losses else np.nan
            td_abs_mean = float(np.mean(td_abs_vals)) if td_abs_vals else np.nan

            row: dict[str, float] = {
                "update": float(update),
                "rollout_reward_mean": float(rollout_stats["rollout_reward_mean"]),
                "rollout_cancel_rate": float(rollout_stats["rollout_cancel_rate"]),
                "q_loss": float(q_loss),
                "td_abs_mean": float(td_abs_mean),
                "epsilon": float(eps),
                "per_beta": float(beta),
                "replay_size": float(self.replay.size),
            }

            run_eval = update == 1 or update % int(self.cfg.eval_every) == 0 or update == int(self.cfg.updates)
            if run_eval:
                train_det = self.evaluate_split(split="train", mode="deterministic")
                test_det = self.evaluate_split(split="test", mode="deterministic")
                row.update(
                    {
                        "train_reward_det": float(train_det["reward_mean"]),
                        "test_reward_det": float(test_det["reward_mean"]),
                        "train_acc_det": float(train_det["decision_accuracy"]),
                        "test_acc_det": float(test_det["decision_accuracy"]),
                        "test_cancel_rate_det": float(test_det["cancel_rate"]),
                        "test_cancel_success_count_det": float(test_det["cancel_success_count"]),
                        "test_cancel_needed_count_det": float(test_det["cancel_needed_count"]),
                        "test_cancel_bad_rate_det": float(test_det["cancel_needed_rate"]),
                    }
                )
                if log_this_update:
                    print(
                        f"[IQN] update={update:4d}/{self.cfg.updates} "
                        f"batch={self.cfg.rollout_size} "
                        f"collect_t={collect_elapsed:.1f}s update_t={update_elapsed:.1f}s "
                        f"rollout_reward={rollout_stats['rollout_reward_mean']:+.4f} "
                        f"q_loss={q_loss:+.4f} td_abs={td_abs_mean:+.4f} "
                        f"eps={eps:.3f} replay={self.replay.size} "
                        f"test_reward_det={test_det['reward_mean']:+.4f} "
                        f"test_acc_det={test_det['decision_accuracy']:.4f} "
                        f"test_cancel_rate={test_det['cancel_rate']:.4f} "
                        f"test_cancel_success={test_det['cancel_success_count']:.0f} "
                        f"test_bad_cancel={test_det['cancel_needed_count']:.0f}"
                    )
            elif log_this_update:
                print(
                    f"[IQN] update={update:4d}/{self.cfg.updates} "
                    f"batch={self.cfg.rollout_size} "
                    f"collect_t={collect_elapsed:.1f}s update_t={update_elapsed:.1f}s "
                    f"rollout_reward={rollout_stats['rollout_reward_mean']:+.4f} "
                    f"q_loss={q_loss:+.4f} td_abs={td_abs_mean:+.4f} "
                    f"eps={eps:.3f} replay={self.replay.size}"
                )

            self.history.append(row)
            executed_updates = update

        elapsed = time.perf_counter() - t0
        print(f"[IQN] Training done in {_format_minutes_seconds(elapsed)}")
        self.training_status["executed_updates"] = int(executed_updates)
        self.training_status["elapsed_s"] = float(elapsed)

        final_metrics = {
            "train_deterministic": self.evaluate_split(split="train", mode="deterministic"),
            "test_deterministic": self.evaluate_split(split="test", mode="deterministic"),
            "train_stochastic": self.evaluate_split(split="train", mode="stochastic"),
            "test_stochastic": self.evaluate_split(split="test", mode="stochastic"),
        }
        return final_metrics

    def save(
        self,
        out_dir: str | Path,
        final_metrics: dict[str, dict[str, float]],
        extra_config: dict | None = None,
    ) -> Path:
        out = Path(out_dir).expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)

        self.q_online.save_weights(str(out / "q_online.weights.h5"))
        self.q_target.save_weights(str(out / "q_target.weights.h5"))
        self.q_policy.save_weights(str(out / "q_policy.weights.h5"))
        save_bundle_metadata(out / "dataset_metadata.json", self.data)
        save_json(
            out / "run_config.json",
            {
                "iqn_config": asdict(self.cfg),
                "reward_config": asdict(self.reward_cfg),
                "extra_config": extra_config or {},
            },
        )
        save_json(out / "final_metrics.json", final_metrics)
        save_json(out / "history.json", self.history)
        save_json(out / "training_status.json", self.training_status)
        generated_plots = self._write_run_plots(
            out,
            final_metrics=final_metrics,
            plot_context=extra_config or {},
        )
        save_json(
            out / "plots_manifest.json",
            {
                "generated": generated_plots,
                "count": int(len(generated_plots)),
            },
        )
        return out


def _parse_hidden_sizes(raw: str) -> tuple[int, ...]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        return (128, 128)
    hidden = tuple(int(x) for x in parts)
    if any(h <= 0 for h in hidden):
        raise ValueError("All hidden sizes must be positive integers.")
    return hidden


def _parse_feature_list(raw: str | None) -> tuple[str, ...] | None:
    if raw is None:
        return None
    cols = tuple(x.strip() for x in raw.split(",") if x.strip())
    return cols or None


def build_arg_parser(
    defaults: dict | None = None,
    default_config: str | None = None,
    default_stage: str | None = None,
    default_trial_index: int | None = None,
) -> argparse.ArgumentParser:
    train_default, test_default = default_dataset_paths()
    p = argparse.ArgumentParser(description="Train IQN agent for truck cancellation.")

    p.add_argument(
        "--config",
        type=str,
        default=default_config if default_config is not None else str(default_config_path()),
        help="Path to config_ppo.json (used for stage/trial sweep defaults)",
    )
    p.add_argument("--stage", type=str, default=default_stage, help="Optional stage name inside config file")
    p.add_argument("--trial-index", type=int, default=default_trial_index, help="Optional trial index in stage grid")
    p.add_argument("--list-trials", action="store_true", help="Print stage trials from config and exit")

    p.add_argument("--train-path", type=str, default=str(train_default), help="Path to df_per_dest_train.pkl")
    p.add_argument("--test-path", type=str, default=str(test_default), help="Path to df_per_dest_test.pkl")
    p.add_argument("--output-dir", type=str, default=None, help="Directory to store IQN artifacts")
    p.add_argument(
        "--output-root",
        type=str,
        default=str(default_output_root_path()),
        help="Root output directory used when --output-dir is not provided",
    )
    p.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Named run folder inside --output-root (example: iqn_run_1)",
    )

    p.add_argument("--state-features", type=str, default=None, help="Comma-separated numeric feature columns")
    p.add_argument("--no-center-onehot", action="store_true", help="Disable center one-hot features")
    p.add_argument("--no-dest-onehot", action="store_true", help="Disable destination one-hot features")
    p.add_argument("--no-normalize", action="store_true", help="Disable numeric z-score normalization")
    p.add_argument(
        "--label-source",
        type=str,
        default="dataset_label",
        help="How to derive needed-truck label: dataset_label|fill_threshold",
    )
    p.add_argument("--needed-fill-threshold", type=float, default=0.2, help="Needed if last truck fill >= threshold")
    p.add_argument("--truck-capacity", type=int, default=100, help="Parcels per truck for fill-ratio criterion")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--updates", type=int, default=300)
    p.add_argument("--rollout-size", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--grad-steps-per-update", type=int, default=4)
    p.add_argument("--min-replay-size", type=int, default=4096)
    p.add_argument("--replay-capacity", type=int, default=200000)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--grad-clip-norm", type=float, default=10.0)
    p.add_argument("--hidden-sizes", type=str, default="128,128")

    p.add_argument("--n-quantiles", type=int, default=64)
    p.add_argument("--n-target-quantiles", type=int, default=64)
    p.add_argument("--n-cos", type=int, default=64)
    p.add_argument("--kappa", type=float, default=1.0)
    p.add_argument("--dueling", dest="dueling", action="store_true")
    p.add_argument("--no-dueling", dest="dueling", action="store_false")
    p.set_defaults(dueling=True)
    p.add_argument("--noisy-nets", action="store_true")

    p.add_argument("--target-update-mode", type=str, default="polyak", help="polyak|hard")
    p.add_argument("--target-update-every", type=int, default=10)
    p.add_argument("--target-update-tau", type=float, default=0.01)

    p.add_argument("--use-per", dest="use_per", action="store_true")
    p.add_argument("--no-per", dest="use_per", action="store_false")
    p.set_defaults(use_per=True)
    p.add_argument("--per-alpha", type=float, default=0.6)
    p.add_argument("--per-beta0", type=float, default=0.4)
    p.add_argument("--per-beta1", type=float, default=1.0)
    p.add_argument("--per-eps", type=float, default=1e-6)

    p.add_argument("--epsilon-start", type=float, default=0.20)
    p.add_argument("--epsilon-end", type=float, default=0.02)
    p.add_argument("--epsilon-decay-updates", type=int, default=200)
    p.add_argument("--decision-temperature", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=10)

    p.add_argument("--reward-keep-needed", type=float, default=1.0)
    p.add_argument("--reward-cancel-not-needed", type=float, default=1.0)
    p.add_argument("--reward-cancel-needed", type=float, default=-5.0)
    p.add_argument("--reward-keep-not-needed", type=float, default=-1.0)
    p.add_argument(
        "--early-cancel-bonus",
        type=float,
        default=0.5,
        help="Extra reward for successful cancellations done earlier in the day",
    )
    p.add_argument(
        "--early-cancel-penalty",
        type=float,
        default=0.5,
        help="Extra penalty for wrong cancellations done earlier in the day",
    )
    if defaults:
        p.set_defaults(**defaults)
    return p


def main() -> None:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=str, default=str(default_config_path()))
    bootstrap.add_argument("--stage", type=str, default=None)
    bootstrap.add_argument("--trial-index", type=int, default=None)
    bootstrap.add_argument("--list-trials", action="store_true")
    pre_args, _unknown = bootstrap.parse_known_args()

    config_path = Path(pre_args.config).expanduser().resolve()
    config_defaults: dict = {}
    config_meta: dict = {"config_path": str(config_path)}

    if config_path.exists():
        config_doc = load_iqn_config(config_path)
        if pre_args.list_trials:
            stage_name, trials, tag_keys = list_stage_trials(config_doc, stage=pre_args.stage)
            print(f"[IQN] config={config_path}")
            if stage_name is None:
                print("[IQN] No stages configured.")
            else:
                print(f"[IQN] stage={stage_name} trials={len(trials)}")
                if tag_keys:
                    print(f"[IQN] tag_keys={tag_keys}")
                for i, trial in enumerate(trials):
                    print(f"  {i}: {trial}")
            return
        config_defaults, resolved = resolve_run_defaults(
            config_doc,
            stage=pre_args.stage,
            trial_index=pre_args.trial_index,
        )
        config_meta.update(resolved)
    elif "--config" in sys.argv:
        raise FileNotFoundError(f"--config path does not exist: {config_path}")

    args = build_arg_parser(
        defaults=config_defaults,
        default_config=str(config_path),
        default_stage=pre_args.stage,
        default_trial_index=pre_args.trial_index,
    ).parse_args()
    hidden_sizes = _parse_hidden_sizes(args.hidden_sizes)
    state_features = _parse_feature_list(args.state_features)
    config_meta.update(_load_dataset_sim_context(args.train_path))

    feature_cfg = FeatureConfig(
        numeric_features=state_features if state_features is not None else FeatureConfig().numeric_features,
        include_center_one_hot=(not args.no_center_onehot),
        include_dest_one_hot=(not args.no_dest_onehot),
        normalize_numeric=(not args.no_normalize),
    )
    label_cfg = LabelConfig(
        source=str(args.label_source),
        needed_fill_threshold=float(args.needed_fill_threshold),
        n_parcels_per_truck=int(args.truck_capacity),
    )

    data = load_data_bundle(
        train_path=args.train_path,
        test_path=args.test_path,
        feature_config=feature_cfg,
        label_config=label_cfg,
    )

    reward_cfg = RewardConfig(
        keep_when_needed=float(args.reward_keep_needed),
        cancel_when_not_needed=float(args.reward_cancel_not_needed),
        cancel_when_needed=float(args.reward_cancel_needed),
        keep_when_not_needed=float(args.reward_keep_not_needed),
        early_cancel_bonus=float(args.early_cancel_bonus),
        early_cancel_penalty=float(args.early_cancel_penalty),
    )

    iqn_cfg = IQNConfig(
        seed=int(args.seed),
        updates=int(args.updates),
        rollout_size=int(args.rollout_size),
        batch_size=int(args.batch_size),
        grad_steps_per_update=int(args.grad_steps_per_update),
        min_replay_size=int(args.min_replay_size),
        replay_capacity=int(args.replay_capacity),
        gamma=float(args.gamma),
        lr=float(args.lr),
        grad_clip_norm=float(args.grad_clip_norm),
        hidden_sizes=hidden_sizes,
        n_quantiles=int(args.n_quantiles),
        n_target_quantiles=int(args.n_target_quantiles),
        n_cos=int(args.n_cos),
        kappa=float(args.kappa),
        dueling=bool(args.dueling),
        noisy_nets=bool(args.noisy_nets),
        target_update_mode=str(args.target_update_mode),
        target_update_every=int(args.target_update_every),
        target_update_tau=float(args.target_update_tau),
        use_per=bool(args.use_per),
        per_alpha=float(args.per_alpha),
        per_beta0=float(args.per_beta0),
        per_beta1=float(args.per_beta1),
        per_eps=float(args.per_eps),
        epsilon_start=float(args.epsilon_start),
        epsilon_end=float(args.epsilon_end),
        epsilon_decay_updates=int(args.epsilon_decay_updates),
        decision_temperature=float(args.decision_temperature),
        eval_every=int(args.eval_every),
    )

    trainer = IQNTruckCancellationOptimiser(data=data, reward_config=reward_cfg, cfg=iqn_cfg)
    final_metrics = trainer.train()

    if args.output_dir is None:
        if args.run_name:
            out_dir = Path(args.output_root).expanduser() / str(args.run_name)
        else:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            out_dir = Path(args.output_root).expanduser() / f"iqn_{timestamp}"
    else:
        out_dir = Path(args.output_dir).expanduser()
    saved_dir = trainer.save(out_dir, final_metrics, extra_config=config_meta)

    print("[IQN] Final metrics:")
    for key, val in final_metrics.items():
        print(
            f"  - {key}: reward={val['reward_mean']:+.4f}, acc={val['decision_accuracy']:.4f}, "
            f"cancel_rate={val['cancel_rate']:.4f}, "
            f"cancel_success={val['cancel_success_count']:.0f}, "
            f"cancel_needed={val['cancel_needed_count']:.0f}"
        )
    print(f"[IQN] Artifacts saved to: {saved_dir}")
    print(f"[IQN] Training status: {saved_dir / 'training_status.json'}")
    print(f"[IQN] Plot manifest: {saved_dir / 'plots_manifest.json'}")


if __name__ == "__main__":
    main()
