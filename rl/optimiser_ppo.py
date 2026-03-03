#!/usr/bin/env python3
"""PPO optimiser for truck-cancellation decisions.

The task is modeled as a one-step MDP:
- state: route/day/hour level features from toy_sim dataset
- action: 0=keep last truck, 1=cancel last truck
- reward: configurable asymmetric matrix (heavy penalty for bad cancellation)

Usage example:
    python optimiser_ppo.py --updates 300 --rollout-size 4096
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
        BinaryPolicyNetwork,
        DataBundle,
        FeatureConfig,
        LabelConfig,
        OneStepCancellationEnv,
        RewardConfig,
        TrajectoryCancellationEnv,
        ValueNetwork,
        apply_gradients_clipped,
        bernoulli_entropy_from_logits,
        bernoulli_log_prob_from_logits,
        choose_actions,
        compute_decision_metrics,
        compute_rewards,
        default_dataset_paths,
        load_data_bundle,
        predict_cancel_probability,
        save_bundle_metadata,
        save_json,
        seed_everything,
    )
    from .config_ppo import (
        default_config_path,
        list_stage_trials,
        load_config as load_ppo_config,
        resolve_run_defaults,
    )
except ImportError:
    from base import (
        BinaryPolicyNetwork,
        DataBundle,
        FeatureConfig,
        LabelConfig,
        OneStepCancellationEnv,
        RewardConfig,
        TrajectoryCancellationEnv,
        ValueNetwork,
        apply_gradients_clipped,
        bernoulli_entropy_from_logits,
        bernoulli_log_prob_from_logits,
        choose_actions,
        compute_decision_metrics,
        compute_rewards,
        default_dataset_paths,
        load_data_bundle,
        predict_cancel_probability,
        save_bundle_metadata,
        save_json,
        seed_everything,
    )
    from config_ppo import (
        default_config_path,
        list_stage_trials,
        load_config as load_ppo_config,
        resolve_run_defaults,
    )


def default_output_root_path() -> Path:
    return Path(__file__).resolve().parent / "outputs"


@dataclass
class PPOConfig:
    """Hyperparameters for PPO training."""

    seed: int = 42
    updates: int = 300
    rollout_size: int = 4096
    ppo_epochs: int = 8
    minibatch_size: int = 512
    clip_ratio: float = 0.2
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    entropy_coef: float = 0.01
    value_coef: float = 1.0
    grad_clip_norm: float = 0.5
    use_critic: bool = True
    reward_baseline_momentum: float = 0.95
    gamma: float = 1.0
    hidden_sizes: tuple[int, ...] = (128, 128)
    normalize_advantages: bool = True
    eval_every: int = 10

    # Decision rules used for evaluation/deployment.
    decision_threshold: float = 0.5
    stochastic_min_prob: float = 0.05
    stochastic_max_prob: float = 0.95

    # Optional convergence-based early stopping.
    early_stop_enabled: bool = False
    early_stop_warmup: int = 100
    early_stop_window: int = 40
    early_stop_check_every: int = 10
    early_stop_patience: int = 3
    early_stop_actor_slope_threshold: float = 1e-4
    early_stop_critic_slope_threshold: float = 5e-4


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


def _linear_slope(values: np.ndarray) -> float:
    """Return least-squares slope of y over index x=0..n-1."""
    y = np.asarray(values, dtype=np.float32).reshape(-1)
    if y.size < 2:
        return 0.0
    x = np.arange(y.size, dtype=np.float32)
    x_centered = x - float(np.mean(x))
    y_centered = y - float(np.mean(y))
    denom = float(np.sum(x_centered * x_centered))
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(x_centered * y_centered) / denom)


def _format_minutes_seconds(seconds: float) -> str:
    total_seconds = max(0, int(round(float(seconds))))
    minutes, rem = divmod(total_seconds, 60)
    return f"{minutes}m {rem:02d}s"


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


class PPOTruckCancellationOptimiser:
    """Offline PPO trainer for the truck-cancellation policy."""

    def __init__(
        self,
        data: DataBundle,
        reward_config: RewardConfig | None = None,
        cfg: PPOConfig | None = None,
    ):
        self.data = data
        self.reward_cfg = reward_config or RewardConfig()
        self.cfg = cfg or PPOConfig()

        self.rng = seed_everything(self.cfg.seed)

        self.policy = BinaryPolicyNetwork(input_dim=self.data.state_dim, hidden_sizes=self.cfg.hidden_sizes)
        self.value_fn = ValueNetwork(input_dim=self.data.state_dim, hidden_sizes=self.cfg.hidden_sizes)
        _ = self.policy(tf.zeros((1, self.data.state_dim), dtype=tf.float32))
        _ = self.value_fn(tf.zeros((1, self.data.state_dim), dtype=tf.float32))

        self.actor_opt = optimizers.Adam(learning_rate=self.cfg.actor_lr)
        self.critic_opt = optimizers.Adam(learning_rate=self.cfg.critic_lr)
        self.moving_reward_baseline: float | None = None

        train_episode_ids = self.data.train_episode_ids
        can_use_trajectory = (
            train_episode_ids is not None
            and train_episode_ids.shape[0] == self.data.x_train.shape[0]
            and int(np.unique(train_episode_ids).size) < int(self.data.x_train.shape[0])
        )
        self.trajectory_mode = bool(can_use_trajectory)
        if self.trajectory_mode:
            self.env = TrajectoryCancellationEnv(
                states=self.data.x_train,
                labels_needed=self.data.y_train,
                episode_ids=np.asarray(train_episode_ids, dtype=np.int32),
                reward_config=self.reward_cfg,
                hours=self.data.train_hours,
                min_hour=self.data.min_hour,
                max_hour=self.data.max_hour,
                gamma=float(self.cfg.gamma),
                rng=self.rng,
            )
        else:
            self.env = OneStepCancellationEnv(
                states=self.data.x_train,
                labels_needed=self.data.y_train,
                reward_config=self.reward_cfg,
                hours=self.data.train_hours,
                min_hour=self.data.min_hour,
                max_hour=self.data.max_hour,
                rng=self.rng,
            )
            print("[PPO] Warning: trajectory grouping unavailable; falling back to one-step sampling.")

        self.history: list[dict] = []
        self.training_status: dict[str, object] = {
            "requested_updates": int(self.cfg.updates),
            "executed_updates": 0,
            "stopped_early": False,
            "stop_reason": "max_updates_reached",
            "early_stop_enabled": bool(self.cfg.early_stop_enabled),
            "use_critic": bool(self.cfg.use_critic),
            "reward_baseline_momentum": float(self.cfg.reward_baseline_momentum),
            "gamma": float(self.cfg.gamma),
            "trajectory_mode": bool(self.trajectory_mode),
            "early_stop_warmup": int(self.cfg.early_stop_warmup),
            "early_stop_window": int(self.cfg.early_stop_window),
            "early_stop_check_every": int(self.cfg.early_stop_check_every),
            "early_stop_patience": int(self.cfg.early_stop_patience),
            "early_stop_actor_slope_threshold": float(self.cfg.early_stop_actor_slope_threshold),
            "early_stop_critic_slope_threshold": float(self.cfg.early_stop_critic_slope_threshold),
            "last_actor_slope": None,
            "last_critic_slope": None,
            "plateau_counter": 0,
        }

    def _ppo_update(
        self,
        rollout,
        update_idx: int | None = None,
        total_updates: int | None = None,
        log_update_progress: bool = True,
    ) -> tuple[float, float, float]:
        idx_all = np.arange(rollout.size, dtype=np.int32)
        actor_losses: list[float] = []
        critic_losses: list[float] = []
        entropies: list[float] = []

        adv = rollout.advantages.astype(np.float32, copy=True)
        if self.cfg.normalize_advantages:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        n_minibatches = int(np.ceil(float(rollout.size) / float(max(1, self.cfg.minibatch_size))))
        total_inner_steps = int(max(1, self.cfg.ppo_epochs * n_minibatches))
        heartbeat_enabled = bool(log_update_progress) and total_inner_steps >= 2000
        heartbeat_every = int(max(1, total_inner_steps // 10))
        inner_step = 0
        t_inner = time.perf_counter()

        if heartbeat_enabled:
            print(
                f"[PPO] inner_loop_start "
                f"(epochs={self.cfg.ppo_epochs}, minibatches_per_epoch={n_minibatches}, "
                f"minibatch_size={self.cfg.minibatch_size}, total_steps={total_inner_steps})"
            )

        for _epoch in range(self.cfg.ppo_epochs):
            self.rng.shuffle(idx_all)
            for start in range(0, rollout.size, self.cfg.minibatch_size):
                stop = min(rollout.size, start + self.cfg.minibatch_size)
                mb = idx_all[start:stop]
                if mb.size == 0:
                    continue

                obs = tf.convert_to_tensor(rollout.states[mb], dtype=tf.float32)
                actions = tf.convert_to_tensor(rollout.actions[mb], dtype=tf.float32)
                old_logp = tf.convert_to_tensor(rollout.old_log_probs[mb], dtype=tf.float32)
                adv_mb = tf.convert_to_tensor(adv[mb], dtype=tf.float32)
                ret_mb = tf.convert_to_tensor(rollout.returns[mb], dtype=tf.float32)

                with tf.GradientTape() as tape_actor:
                    logits = tf.squeeze(self.policy(obs, training=True), axis=-1)
                    logp = bernoulli_log_prob_from_logits(logits, actions)
                    ratio = tf.exp(logp - old_logp)
                    unclipped = ratio * adv_mb
                    clipped = tf.clip_by_value(
                        ratio,
                        1.0 - float(self.cfg.clip_ratio),
                        1.0 + float(self.cfg.clip_ratio),
                    ) * adv_mb
                    entropy = tf.reduce_mean(bernoulli_entropy_from_logits(logits))
                    actor_loss = -tf.reduce_mean(tf.minimum(unclipped, clipped))
                    actor_loss = actor_loss - float(self.cfg.entropy_coef) * entropy

                apply_gradients_clipped(
                    optimizer=self.actor_opt,
                    loss=actor_loss,
                    variables=self.policy.trainable_variables,
                    tape=tape_actor,
                    grad_clip_norm=self.cfg.grad_clip_norm,
                )

                if bool(self.cfg.use_critic):
                    with tf.GradientTape() as tape_critic:
                        values = tf.squeeze(self.value_fn(obs, training=True), axis=-1)
                        critic_loss = float(self.cfg.value_coef) * tf.reduce_mean(tf.square(ret_mb - values))

                    apply_gradients_clipped(
                        optimizer=self.critic_opt,
                        loss=critic_loss,
                        variables=self.value_fn.trainable_variables,
                        tape=tape_critic,
                        grad_clip_norm=self.cfg.grad_clip_norm,
                    )
                    critic_loss_value = float(critic_loss.numpy())
                else:
                    critic_loss_value = 0.0

                actor_losses.append(float(actor_loss.numpy()))
                critic_losses.append(critic_loss_value)
                entropies.append(float(entropy.numpy()))
                inner_step += 1

                if heartbeat_enabled and (inner_step % heartbeat_every == 0 or inner_step == total_inner_steps):
                    elapsed = time.perf_counter() - t_inner
                    pct = 100.0 * float(inner_step) / float(total_inner_steps)
                    print(
                        f"[PPO] inner_progress={inner_step}/{total_inner_steps} "
                        f"({pct:.1f}%) elapsed={_format_minutes_seconds(elapsed)}"
                    )

        return (
            float(np.mean(actor_losses)) if actor_losses else 0.0,
            float(np.mean(critic_losses)) if critic_losses else 0.0,
            float(np.mean(entropies)) if entropies else 0.0,
        )

    def _split_arrays(
        self,
        split: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        split_norm = str(split).strip().lower()
        if split_norm == "train":
            x = self.data.x_train
            y = self.data.y_train
            h = self.data.train_hours
            episode_ids = self.data.train_episode_ids
        elif split_norm == "test":
            x = self.data.x_test
            y = self.data.y_test
            h = self.data.test_hours
            episode_ids = self.data.test_episode_ids
        else:
            raise ValueError("split must be train|test")
        hours = np.asarray(h, dtype=np.int32) if h is not None else np.zeros((x.shape[0],), dtype=np.int32)
        if episode_ids is None or int(np.asarray(episode_ids).shape[0]) != int(x.shape[0]):
            ep = np.arange(x.shape[0], dtype=np.int32)
        else:
            ep = np.asarray(episode_ids, dtype=np.int32)
        return (
            np.asarray(x, dtype=np.float32),
            (np.asarray(y, dtype=np.float32) >= 0.5).astype(np.int32),
            hours,
            ep,
        )

    def _simulate_split_decisions(
        self,
        split: str = "test",
        mode: str = "deterministic",
    ) -> dict[str, np.ndarray]:
        x, needed_all, hours_all, episode_ids = self._split_arrays(split=split)
        cancel_prob_all = predict_cancel_probability(self.policy, x)

        # One-step fallback: row-wise decisions.
        if not bool(self.trajectory_mode):
            actions = choose_actions(
                cancel_prob=cancel_prob_all,
                mode=mode,
                threshold=float(self.cfg.decision_threshold),
                rng=self.rng,
                min_prob=float(self.cfg.stochastic_min_prob),
                max_prob=float(self.cfg.stochastic_max_prob),
            ).astype(np.int32)
            return {
                "row_indices": np.arange(x.shape[0], dtype=np.int32),
                "cancel_prob": np.asarray(cancel_prob_all, dtype=np.float32),
                "actions": actions,
                "needed": needed_all.astype(np.int32),
                "hours": hours_all.astype(np.int32),
            }

        order = np.lexsort((hours_all.astype(np.float32), episode_ids.astype(np.int64)))
        ep_sorted = episode_ids[order]
        split_points = np.flatnonzero(np.diff(ep_sorted)) + 1
        episode_rows = np.split(order, split_points)

        min_prob = float(self.cfg.stochastic_min_prob)
        max_prob = float(self.cfg.stochastic_max_prob)
        lo = float(max(0.0, min(1.0, min_prob)))
        hi = float(max(0.0, min(1.0, max_prob)))
        if hi < lo:
            lo, hi = hi, lo

        row_indices: list[int] = []
        cancel_prob: list[float] = []
        actions: list[int] = []
        needed: list[int] = []
        hours: list[int] = []

        mode_norm = str(mode).strip().lower()
        for rows in episode_rows:
            if rows.size == 0:
                continue
            for t, idx_row in enumerate(rows.tolist()):
                p = float(cancel_prob_all[idx_row])
                if mode_norm == "deterministic":
                    a = 1 if p >= float(self.cfg.decision_threshold) else 0
                elif mode_norm == "stochastic":
                    pp = float(np.clip(p, lo, hi))
                    a = int(self.rng.binomial(1, pp))
                else:
                    raise ValueError(f"Unknown action mode: {mode!r}. Expected deterministic|stochastic.")

                row_indices.append(int(idx_row))
                cancel_prob.append(float(p))
                actions.append(int(a))
                needed.append(int(needed_all[idx_row]))
                hours.append(int(hours_all[idx_row]))

                is_last = t == (rows.size - 1)
                if bool(a == 1) or is_last:
                    break

        return {
            "row_indices": np.asarray(row_indices, dtype=np.int32),
            "cancel_prob": np.asarray(cancel_prob, dtype=np.float32),
            "actions": np.asarray(actions, dtype=np.int32),
            "needed": np.asarray(needed, dtype=np.int32),
            "hours": np.asarray(hours, dtype=np.int32),
        }

    def evaluate_split(
        self,
        split: str = "test",
        mode: str = "deterministic",
    ) -> dict[str, float]:
        details = self._simulate_split_decisions(split=split, mode=mode)
        actions = details["actions"]
        needed = details["needed"]
        hours = details["hours"]
        cancel_prob = details["cancel_prob"]
        rewards = compute_rewards(
            actions,
            needed,
            self.reward_cfg,
            hours=hours,
            min_hour=self.data.min_hour,
            max_hour=self.data.max_hour,
        )
        metrics = compute_decision_metrics(actions, needed, rewards)
        metrics["avg_cancel_probability"] = float(np.mean(cancel_prob)) if cancel_prob.size else 0.0
        metrics["n_decision_points"] = float(actions.shape[0])
        return metrics

    def _predict_split_details(
        self,
        split: str = "test",
        mode: str = "deterministic",
    ) -> dict[str, np.ndarray]:
        return self._simulate_split_decisions(split=split, mode=mode)

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

        def history_cancel_success_rate_series() -> np.ndarray:
            vals: list[float] = []
            for row in self.history:
                succ_raw = row.get("test_cancel_success_count_det")
                bad_raw = row.get("test_cancel_needed_count_det")
                if succ_raw is None or bad_raw is None:
                    vals.append(np.nan)
                    continue
                try:
                    succ = float(succ_raw)
                    bad = float(bad_raw)
                except Exception:
                    vals.append(np.nan)
                    continue
                denom = succ + bad
                vals.append(float(succ / denom) if denom > 1e-12 else np.nan)
            return np.asarray(vals, dtype=np.float32)

        def finite_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            mask = np.isfinite(np.asarray(y, dtype=np.float32))
            return np.asarray(x)[mask], np.asarray(y, dtype=np.float32)[mask]

        def cumulative_recall_from_event_hours(
            hour_axis_vals: np.ndarray,
            tp_event_hours: np.ndarray,
            n_positive: int,
        ) -> np.ndarray:
            """Monotonic cumulative recall: fraction of positives cancelled by each hour."""
            y = np.full(np.asarray(hour_axis_vals).shape, np.nan, dtype=np.float32)
            if int(n_positive) <= 0:
                return y
            y.fill(0.0)
            ev = np.asarray(tp_event_hours, dtype=np.float64)
            ev = ev[np.isfinite(ev)]
            if ev.size <= 0:
                return y
            ev_sorted = np.sort(ev)
            h = np.asarray(hour_axis_vals, dtype=np.float64)
            cum = np.searchsorted(ev_sorted, h, side="right").astype(np.float64)
            return (cum / float(n_positive)).astype(np.float32)

        try:
            updates = np.asarray([int(row.get("update", 0)) for row in self.history], dtype=np.int32)
            if updates.size > 0:
                fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
                axes[0, 0].plot(updates, history_series("rollout_reward_mean"), linewidth=1.2)
                axes[0, 0].set_title("Rollout Reward Mean")
                axes[0, 0].set_ylabel("reward")
                axes[0, 0].grid(alpha=0.2)

                test_reward_series = history_series("test_reward_det")
                u_reward, y_reward = finite_xy(updates, test_reward_series)
                if y_reward.size > 0:
                    axes[0, 1].plot(u_reward, y_reward, linewidth=1.2, color="tab:orange", marker="o", markersize=3)
                axes[0, 1].set_title("Test Deterministic Reward")
                axes[0, 1].set_ylabel("reward")
                axes[0, 1].grid(alpha=0.2)

                test_acc_series = history_series("test_acc_det")
                u_acc, y_acc = finite_xy(updates, test_acc_series)
                if y_acc.size > 0:
                    axes[1, 0].plot(u_acc, y_acc, linewidth=1.2, color="tab:green", marker="o", markersize=3)
                axes[1, 0].set_title("Test Deterministic Accuracy")
                axes[1, 0].set_xlabel("update")
                axes[1, 0].set_ylabel("accuracy")
                axes[1, 0].set_ylim(-0.02, 1.02)
                axes[1, 0].grid(alpha=0.2)

                axes[1, 1].plot(updates, history_series("entropy"), linewidth=1.0, color="tab:purple")
                axes[1, 1].set_title("Policy Entropy")
                axes[1, 1].set_xlabel("update")
                axes[1, 1].set_ylabel("entropy")
                axes[1, 1].grid(alpha=0.2)

                fig.tight_layout()
                fig.savefig(plot_dir / "training_curves.png", dpi=140)
                plt.close(fig)
                generated.append("plots/training_curves.png")

                actor_loss_series = history_series("actor_loss")
                u_actor, y_actor = finite_xy(updates, actor_loss_series)
                if y_actor.size > 0:
                    fig, ax = plt.subplots(figsize=(9, 4))
                    ax.plot(u_actor, y_actor, linewidth=1.2, color="tab:blue")
                    ax.set_xlabel("update")
                    ax.set_ylabel("actor_loss")
                    ax.set_title("Actor Loss over Updates")
                    ax.grid(alpha=0.2)
                    fig.tight_layout()
                    fig.savefig(plot_dir / "actor_loss_over_updates.png", dpi=140)
                    plt.close(fig)
                    generated.append("plots/actor_loss_over_updates.png")

                critic_loss_series = history_series("critic_loss")
                u_critic, y_critic = finite_xy(updates, critic_loss_series)
                if y_critic.size > 0:
                    fig, ax = plt.subplots(figsize=(9, 4))
                    ax.plot(u_critic, y_critic, linewidth=1.2, color="tab:red")
                    ax.set_xlabel("update")
                    ax.set_ylabel("critic_loss")
                    ax.set_title("Critic Loss over Updates")
                    ax.grid(alpha=0.2)
                    fig.tight_layout()
                    fig.savefig(plot_dir / "critic_loss_over_updates.png", dpi=140)
                    plt.close(fig)
                    generated.append("plots/critic_loss_over_updates.png")

                cancel_rate_hist = history_series("test_cancel_rate_det")
                cancel_success_rate_hist = history_cancel_success_rate_series()
                u_cancel_rate, y_cancel_rate = finite_xy(updates, cancel_rate_hist)
                u_cancel_succ, y_cancel_succ = finite_xy(updates, cancel_success_rate_hist)
                fig, ax = plt.subplots(figsize=(10, 4.5))
                if y_cancel_rate.size > 0:
                    ax.plot(
                        u_cancel_rate,
                        y_cancel_rate,
                        linewidth=1.4,
                        marker="o",
                        markersize=2.5,
                        label="cancel_rate_test",
                    )
                if y_cancel_succ.size > 0:
                    ax.plot(
                        u_cancel_succ,
                        y_cancel_succ,
                        linewidth=1.4,
                        marker="o",
                        markersize=2.5,
                        label="cancel_success_rate_test",
                        color="tab:green",
                    )
                ax.set_xlabel("update")
                ax.set_ylabel("rate")
                ax.set_ylim(-0.02, 1.02)
                ax.set_title("Cancel Behavior over Training Updates (Deterministic Test)")
                ax.grid(alpha=0.2)
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(loc="lower right")

                if final_metrics is not None:
                    test_det = final_metrics.get("test_deterministic", {}) if isinstance(final_metrics, dict) else {}
                    if isinstance(test_det, dict):
                        final_lines: list[float] = []
                        try:
                            test_cancel_rate = test_det.get("cancel_rate")
                            if test_cancel_rate is not None:
                                final_lines.append(float(test_cancel_rate))
                        except Exception:
                            pass
                        try:
                            test_cancel_success_rate = test_det.get("cancel_success_rate_among_cancellations")
                            if test_cancel_success_rate is not None:
                                final_lines.append(float(test_cancel_success_rate))
                        except Exception:
                            pass

                        for i, y_val in enumerate(final_lines):
                            ax.axhline(y_val, color="gray", linestyle=":", linewidth=1.2, alpha=0.9)
                            y_offset = 0
                            if len(final_lines) >= 2 and abs(final_lines[0] - final_lines[1]) < 0.03:
                                y_offset = -7 if i == 0 else 7
                            ax.annotate(
                                f"{y_val:.3f}",
                                xy=(0.0, y_val),
                                xycoords=("axes fraction", "data"),
                                xytext=(3, y_offset),
                                textcoords="offset points",
                                ha="left",
                                va="center",
                                color="gray",
                                fontsize=8,
                            )

                fig.tight_layout()
                fig.savefig(plot_dir / "cancel_behavior_over_updates.png", dpi=140)
                plt.close(fig)
                generated.append("plots/cancel_behavior_over_updates.png")
        except Exception:
            pass

        try:
            pred_test = self._predict_split_details(split="test", mode="deterministic")
            test_hours = pred_test["hours"]
            test_actions = pred_test["actions"]
            test_needed = pred_test["needed"]
            test_row_indices = np.asarray(
                pred_test.get("row_indices", np.arange(test_actions.shape[0], dtype=np.int32)),
                dtype=np.int32,
            )
            test_episode_ids_dec: np.ndarray = np.arange(test_actions.shape[0], dtype=np.int64)
            test_episode_ids_full = self.data.test_episode_ids
            if (
                test_episode_ids_full is not None
                and test_row_indices.shape[0] == test_actions.shape[0]
                and test_row_indices.size > 0
            ):
                ep_full = np.asarray(test_episode_ids_full, dtype=np.int64)
                idx_valid_ep = (test_row_indices >= 0) & (test_row_indices < ep_full.shape[0])
                if np.all(idx_valid_ep):
                    test_episode_ids_dec = ep_full[test_row_indices].astype(np.int64)

            test_df_full = self.data.test_df
            test_df_dec: pd.DataFrame | None = None
            if (
                test_df_full is not None
                and hasattr(test_df_full, "columns")
                and test_row_indices.shape[0] == test_actions.shape[0]
                and test_row_indices.size > 0
            ):
                idx_valid = (test_row_indices >= 0) & (test_row_indices < len(test_df_full))
                if np.all(idx_valid):
                    test_df_dec = test_df_full.iloc[test_row_indices].reset_index(drop=True)

            train_hours = (
                np.asarray(self.data.train_hours, dtype=np.int32)
                if self.data.train_hours is not None
                else np.zeros((self.data.x_train.shape[0],), dtype=np.int32)
            )
            hour_axis = _build_hour_axis(train_hours, test_hours)
            train_counts = _hourly_count(hour_axis, train_hours)
            test_counts = _hourly_count(hour_axis, test_hours)
            # Example day-route trajectory: cumulative volume + agent vs optimal decisions by hour.
            example_plotted = False
            if (
                test_df_dec is not None
                and all(col in test_df_dec.columns for col in ("center", "dest", "hour"))
                and test_actions.shape[0] == len(test_df_dec)
            ):
                center_vals = np.asarray(test_df_dec["center"].astype(str).to_numpy())
                dest_vals = np.asarray(test_df_dec["dest"].astype(str).to_numpy())
                hour_vals = np.asarray(test_df_dec["hour"].to_numpy(), dtype=np.int32)
                uniq_hours = np.sort(np.unique(hour_vals))

                route_pairs = sorted(set(zip(center_vals.tolist(), dest_vals.tolist())))
                example_idx: np.ndarray | None = None
                example_center = ""
                example_dest = ""
                example_slot = 0
                example_slot_total = 0

                for center_key, dest_key in route_pairs:
                    route_mask = (center_vals == center_key) & (dest_vals == dest_key)
                    idx_by_hour: list[np.ndarray] = []
                    min_len: int | None = None
                    for h in uniq_hours.tolist():
                        idx_h = np.where(route_mask & (hour_vals == int(h)))[0]
                        if idx_h.size == 0:
                            idx_by_hour = []
                            break
                        idx_h = np.sort(idx_h)
                        idx_by_hour.append(idx_h)
                        min_len = int(idx_h.size) if min_len is None else min(int(min_len), int(idx_h.size))
                    if not idx_by_hour or min_len is None or min_len <= 0:
                        continue

                    slot = int(min_len // 2)
                    example_idx = np.asarray([arr[slot] for arr in idx_by_hour], dtype=np.int32)
                    example_center = center_key
                    example_dest = dest_key
                    example_slot = slot
                    example_slot_total = int(min_len)
                    break

                if example_idx is not None and example_idx.size > 1:
                    ex_hours = hour_vals[example_idx]
                    order = np.argsort(ex_hours)
                    ex_idx = example_idx[order]
                    ex_hours = ex_hours[order]

                    if "max" in test_df_dec.columns:
                        ex_cum = np.asarray(test_df_dec["max"].to_numpy(dtype=np.float32))[ex_idx]
                        cum_label = "cumulative_volume (feature=max)"
                    elif "mean" in test_df_dec.columns:
                        ex_cum = np.asarray(test_df_dec["mean"].to_numpy(dtype=np.float32))[ex_idx]
                        cum_label = "volume_proxy (feature=mean)"
                    elif "delta" in test_df_dec.columns:
                        ex_delta = np.asarray(test_df_dec["delta"].to_numpy(dtype=np.float32))[ex_idx]
                        ex_cum = np.cumsum(ex_delta, dtype=np.float32)
                        cum_label = "cumulative_volume_proxy (cumsum(delta))"
                    else:
                        ex_cum = np.asarray(ex_hours, dtype=np.float32)
                        cum_label = "hour_index_proxy"

                    ex_actions = test_actions[ex_idx].astype(np.int32)
                    ex_needed = test_needed[ex_idx].astype(np.int32)
                    ex_true_action = (ex_needed == 0).astype(np.int32)  # cancel when not needed, keep when needed
                    ex_correct = ex_actions == ex_true_action

                    fig, axes = plt.subplots(
                        2,
                        1,
                        figsize=(10, 7),
                        sharex=True,
                        gridspec_kw={"height_ratios": [2.2, 1.0]},
                    )

                    axes[0].plot(ex_hours, ex_cum, linewidth=1.6, marker="o", color="tab:blue", label=cum_label)
                    axes[0].set_ylabel("volume")
                    axes[0].set_title(
                        "Example Day Trajectory: "
                        f"{example_center}->{example_dest} (sample {example_slot + 1}/{max(1, example_slot_total)})"
                    )
                    axes[0].grid(alpha=0.2)
                    axes[0].legend(loc="upper left")

                    axes[1].step(
                        ex_hours,
                        ex_true_action,
                        where="mid",
                        linewidth=1.4,
                        linestyle="--",
                        color="tab:gray",
                        label="optimal_action",
                    )
                    axes[1].step(
                        ex_hours,
                        ex_actions,
                        where="mid",
                        linewidth=1.4,
                        color="tab:blue",
                        label="agent_action",
                    )
                    for h, a, ok in zip(ex_hours.tolist(), ex_actions.tolist(), ex_correct.tolist()):
                        axes[1].scatter(
                            [h],
                            [a],
                            s=45,
                            c=("tab:green" if ok else "tab:red"),
                            zorder=3,
                        )

                    axes[1].set_yticks([0, 1])
                    axes[1].set_yticklabels(["keep", "cancel"])
                    axes[1].set_xlabel("hour")
                    axes[1].set_ylabel("decision")
                    axes[1].set_ylim(-0.2, 1.2)
                    axes[1].grid(alpha=0.2)
                    axes[1].legend(loc="upper right")

                    final_true = "cancel" if int(ex_true_action[-1]) == 1 else "keep"
                    final_agent = "cancel" if int(ex_actions[-1]) == 1 else "keep"
                    final_ok = bool(ex_correct[-1])
                    axes[0].text(
                        0.99,
                        0.04,
                        f"Final optimal: {final_true}\nFinal agent: {final_agent}\nFinal correct: {final_ok}",
                        transform=axes[0].transAxes,
                        ha="right",
                        va="bottom",
                        fontsize=9,
                        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
                    )

                    fig.tight_layout()
                    fig.savefig(plot_dir / "hourly_volume_profile.png", dpi=140)
                    plt.close(fig)
                    generated.append("plots/hourly_volume_profile.png")
                    example_plotted = True

            if not example_plotted:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(hour_axis, test_counts, marker="o", linewidth=1.2, label="test_decision_points")
                ax.plot(hour_axis, train_counts, marker="o", linewidth=1.2, label="train_decision_points")
                ax.set_xlabel("hour")
                ax.set_ylabel("count")
                ax.set_title("Decision-Point Counts by Hour (fallback)")
                ax.grid(alpha=0.2)
                ax.legend(loc="best")
                fig.tight_layout()
                fig.savefig(plot_dir / "hourly_volume_profile.png", dpi=140)
                plt.close(fig)
                generated.append("plots/hourly_volume_profile.png")

            test_hour_axis = np.sort(np.unique(np.asarray(test_hours, dtype=np.int32)))
            cancel_rate_hour: list[float] = []
            cancel_success_rate_hour: list[float] = []
            for h in test_hour_axis.tolist():
                mask_h = np.asarray(test_hours, dtype=np.int32) == int(h)
                n_h = int(np.sum(mask_h))
                if n_h <= 0:
                    cancel_rate_hour.append(np.nan)
                    cancel_success_rate_hour.append(np.nan)
                    continue
                actions_h = test_actions[mask_h]
                needed_h = test_needed[mask_h]
                cancel_count_h = int(np.sum(actions_h == 1))
                success_count_h = int(np.sum((actions_h == 1) & (needed_h == 0)))
                cancel_rate_hour.append(float(cancel_count_h / n_h))
                cancel_success_rate_hour.append(float(success_count_h / cancel_count_h) if cancel_count_h > 0 else np.nan)

            x = np.arange(test_hour_axis.shape[0], dtype=np.float32)
            width = 0.38
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(x - width / 2.0, cancel_rate_hour, width=width, label="cancel_rate", color="tab:blue")
            ax.bar(
                x + width / 2.0,
                cancel_success_rate_hour,
                width=width,
                label="cancel_success_rate_among_cancellations",
                color="tab:green",
            )
            ax.set_xticks(x)
            ax.set_xticklabels([str(int(h)) for h in test_hour_axis.tolist()])
            ax.set_xlabel("hour")
            ax.set_ylabel("rate")
            ax.set_ylim(-0.02, 1.02)
            ax.set_title("Test: Cancel Rate and Cancel Success Rate by Hour")
            ax.grid(axis="y", alpha=0.2)
            ax.legend(loc="upper right")
            fig.tight_layout()
            fig.savefig(plot_dir / "hourly_decision_rates_test.png", dpi=140)
            plt.close(fig)
            generated.append("plots/hourly_decision_rates_test.png")

            if test_df_dec is not None and "dest" in test_df_dec.columns:
                dest_values = np.asarray(test_df_dec["dest"].astype(str).to_numpy()).reshape(-1)
                if dest_values.shape[0] == test_actions.shape[0]:
                    unique_dest = np.asarray(sorted(np.unique(dest_values)))
                    cancel_rate_by_dest: list[float] = []
                    cancel_success_rate_by_dest: list[float] = []
                    for dest in unique_dest.tolist():
                        mask = dest_values == dest
                        n_rows = int(np.sum(mask))
                        if n_rows <= 0:
                            cancel_rate_by_dest.append(np.nan)
                            cancel_success_rate_by_dest.append(np.nan)
                            continue
                        actions_d = test_actions[mask]
                        needed_d = test_needed[mask]
                        cancel_count = int(np.sum(actions_d == 1))
                        success_count = int(np.sum((actions_d == 1) & (needed_d == 0)))
                        cancel_rate_by_dest.append(float(cancel_count / n_rows))
                        cancel_success_rate_by_dest.append(float(success_count / cancel_count) if cancel_count > 0 else 0.0)

                    x = np.arange(unique_dest.shape[0], dtype=np.float32)
                    width = 0.4
                    fig, ax = plt.subplots(figsize=(11, 4.8))
                    ax.bar(x - width / 2.0, cancel_rate_by_dest, width=width, label="cancel_rate", color="tab:blue")
                    ax.bar(
                        x + width / 2.0,
                        cancel_success_rate_by_dest,
                        width=width,
                        label="cancel_success_rate",
                        color="tab:green",
                    )
                    ax.set_xticks(x)
                    ax.set_xticklabels(unique_dest.tolist())
                    ax.set_xlabel("destination")
                    ax.set_ylabel("rate")
                    ax.set_ylim(-0.02, 1.02)
                    ax.set_title("Test Cancel Metrics by Destination (Deterministic)")
                    ax.grid(axis="y", alpha=0.2)
                    ax.legend(loc="upper right")
                    fig.tight_layout()
                    fig.savefig(plot_dir / "cancel_metrics_by_destination_test.png", dpi=140)
                    plt.close(fig)
                    generated.append("plots/cancel_metrics_by_destination_test.png")

            # Per-trial grid: one panel per route (center -> dest), curve over hour.
            if (
                test_df_dec is not None
                and all(col in test_df_dec.columns for col in ("center", "dest", "hour"))
                and len(test_df_dec) == test_actions.shape[0]
            ):
                center_values = np.asarray(test_df_dec["center"].astype(str).to_numpy())
                dest_values = np.asarray(test_df_dec["dest"].astype(str).to_numpy())
                hour_values = np.asarray(test_df_dec["hour"].to_numpy(), dtype=np.int32)

                eval_df = pd.DataFrame(
                    {
                        "episode_id": test_episode_ids_dec,
                        "center": center_values,
                        "dest": dest_values,
                        "hour": hour_values,
                        "action": test_actions.astype(np.int32),
                        "needed": test_needed.astype(np.int32),
                    }
                )
                eval_df["cancelled"] = (eval_df["action"] == 1).astype(np.int32)
                eval_df["correct_cancel"] = (
                    (eval_df["action"] == 1) & (eval_df["needed"] == 0)
                ).astype(np.int32)
                eval_df["tp_cancel"] = (
                    (eval_df["action"] == 1) & (eval_df["needed"] == 0)
                ).astype(np.int32)
                eval_df["fp_cancel"] = (
                    (eval_df["action"] == 1) & (eval_df["needed"] == 1)
                ).astype(np.int32)
                eval_df["fn_cancel"] = (
                    (eval_df["action"] == 0) & (eval_df["needed"] == 0)
                ).astype(np.int32)
                episode_scope = (
                    eval_df.groupby(["episode_id", "center", "dest"], as_index=False)
                    .agg(
                        should_cancel=(
                            "needed",
                            lambda s: int(np.any(np.asarray(s, dtype=np.int32) == 0)),
                        ),
                    )
                    .reset_index(drop=True)
                )
                tp_events = (
                    eval_df.loc[
                        (eval_df["action"] == 1) & (eval_df["needed"] == 0),
                        ["episode_id", "hour"],
                    ]
                    .sort_values("hour", kind="stable")
                    .drop_duplicates("episode_id", keep="first")
                    .rename(columns={"hour": "tp_hour"})
                    .reset_index(drop=True)
                )
                episode_scope = episode_scope.merge(tp_events, on="episode_id", how="left")

                agg = (
                    eval_df.groupby(["center", "dest", "hour"], as_index=False)
                    .agg(
                        total_trucks=("action", "size"),
                        cancellations=("cancelled", "sum"),
                        correct_cancels=("correct_cancel", "sum"),
                        tp_cancel=("tp_cancel", "sum"),
                        fp_cancel=("fp_cancel", "sum"),
                        fn_cancel=("fn_cancel", "sum"),
                    )
                    .sort_values(["center", "dest", "hour"], kind="stable")
                    .reset_index(drop=True)
                )
                agg["fraction_correctly_cancelled"] = np.where(
                    (agg["tp_cancel"] + agg["fp_cancel"]) > 0,
                    agg["tp_cancel"] / (agg["tp_cancel"] + agg["fp_cancel"]),
                    np.nan,
                )

                # Integrated hourly statistics across all routes:
                # - PPV averaged over (center, dest) lanes per hour
                # - cumulative TPR by hour (episode-level)
                # - cancellation-hour distribution normalized to sum to 1
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
                        cancellations=("cancellations", "sum"),
                    )
                    .sort_values("hour", kind="stable")
                    .reset_index(drop=True)
                )

                if not integrated.empty:
                    total_cancellations = float(
                        np.sum(integrated["cancellations"].to_numpy(dtype=np.float64))
                    )
                    cancel_share = np.zeros((integrated.shape[0],), dtype=np.float64)
                    if total_cancellations > 0.0:
                        cancel_share = (
                            integrated["cancellations"].to_numpy(dtype=np.float64) / total_cancellations
                        )

                    x_hour_integrated = integrated["hour"].to_numpy(dtype=np.float32)
                    y_ppv_integrated = integrated["ppv_mean"].to_numpy(dtype=np.float32)
                    positives_all = episode_scope[
                        episode_scope["should_cancel"].astype(np.int32) == 1
                    ]
                    y_tpr_integrated = cumulative_recall_from_event_hours(
                        hour_axis_vals=x_hour_integrated,
                        tp_event_hours=positives_all["tp_hour"].to_numpy(dtype=np.float64),
                        n_positive=int(positives_all.shape[0]),
                    )

                    fig_int, ax_int = plt.subplots(figsize=(9.2, 4.8))
                    ax_int.plot(
                        x_hour_integrated,
                        y_ppv_integrated,
                        marker="o",
                        linewidth=1.35,
                        linestyle="--",
                        color="tab:blue",
                        markersize=3.2,
                        label="PPV",
                    )
                    ax_int.plot(
                        x_hour_integrated,
                        y_tpr_integrated,
                        marker="o",
                        linewidth=1.2,
                        linestyle="-",
                        color="darkorange",
                        markersize=3.0,
                        label="TPR (cumulative <= hour)",
                    )
                    ax_int.plot(
                        x_hour_integrated,
                        cancel_share,
                        marker="o",
                        linewidth=1.2,
                        linestyle="-",
                        color="tab:red",
                        markersize=2.8,
                        label="cancelled share by hour",
                    )
                    ax_int.set_xlabel("hour")
                    ax_int.set_ylabel("fraction")
                    ax_int.set_ylim(0.0, 1.0)
                    ax_int.grid(alpha=0.2)
                    ax_int.legend(loc="lower right", fontsize=9, frameon=False)

                    int_title = "Test: Integrated Hourly PPV, Cumulative TPR, and Cancellation Share"
                    corr_label = _extract_corr_label(plot_context)
                    if corr_label is not None:
                        int_title = f"{int_title} | {corr_label}"
                    ax_int.set_title(int_title, fontsize=12)
                    fig_int.tight_layout()
                    out_int_pdf = plot_dir / "integrated_statistics_per_hour.pdf"
                    fig_int.savefig(out_int_pdf)
                    plt.close(fig_int)
                    generated.append(f"plots/{out_int_pdf.name}")

                centers_present = sorted(set(agg["center"].astype(str).tolist()))
                if centers_present:
                    corr_label = _extract_corr_label(plot_context)
                    all_dests_present = sorted(set(agg["dest"].astype(str).tolist()))
                    row_destinations: list[list[str]] = []
                    for c in centers_present:
                        default_dests = [d for d in all_dests_present if d != c]
                        center_dests = set(
                            agg.loc[agg["center"].astype(str) == c, "dest"].astype(str).tolist()
                        )
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

                                x_hour = lane["hour"].to_numpy(dtype=np.float32)
                                y_frac = lane["fraction_correctly_cancelled"].to_numpy(dtype=np.float32)
                                ax.plot(
                                    x_hour,
                                    y_frac,
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
                                    ax.set_ylabel(f"{center_key}\ncorrect_frac", fontsize=8)
                                ax.tick_params(axis="both", labelsize=7)

                        fig.suptitle("Test: Hourly Fraction Correctly Cancelled by Route", fontsize=12)
                        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
                        fig.savefig(plot_dir / "hourly_correctly_cancelled_grid_test.png", dpi=140)
                        plt.close(fig)
                        generated.append("plots/hourly_correctly_cancelled_grid_test.png")

                    # One PDF per center with destination subplots.
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
                            denom_ppv = tp_hour + fp_hour
                            y_ppv = np.full_like(tp_hour, np.nan, dtype=np.float32)
                            np.divide(tp_hour, denom_ppv, out=y_ppv, where=denom_ppv > 0.0)
                            lane_pos = episode_scope[
                                (episode_scope["center"].astype(str) == str(center_key))
                                & (episode_scope["dest"].astype(str) == str(dest_key))
                                & (episode_scope["should_cancel"].astype(np.int32) == 1)
                            ]
                            y_tpr = cumulative_recall_from_event_hours(
                                hour_axis_vals=x_hour,
                                tp_event_hours=lane_pos["tp_hour"].to_numpy(dtype=np.float64),
                                n_positive=int(lane_pos.shape[0]),
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
                        center_token = "".join(
                            ch for ch in str(center_key) if ch.isalnum() or ch in {"_", "-"}
                        )
                        if not center_token:
                            center_token = "unknown"
                        out_pdf = plot_dir / f"center_{center_token}_dest_correctly_cancelled_trucks_per_hour.pdf"
                        fig_center.savefig(out_pdf)
                        plt.close(fig_center)
                        generated.append(f"plots/{out_pdf.name}")
        except Exception:
            pass

        return generated

    def train(self) -> dict[str, dict[str, float]]:
        n_train_episodes = int(getattr(self.env, "n_episodes", 0))
        print(
            f"[PPO] Start training: state_dim={self.data.state_dim}, "
            f"n_train={self.data.x_train.shape[0]}, n_test={self.data.x_test.shape[0]}, "
            f"trajectory_mode={self.trajectory_mode}, n_train_episodes={n_train_episodes}"
        )

        t0 = time.perf_counter()
        actor_loss_trace: list[float] = []
        critic_loss_trace: list[float] = []
        plateau_counter = 0
        stop_reason = "max_updates_reached"
        stopped_early = False
        executed_updates = 0

        warmup = max(1, int(self.cfg.early_stop_warmup))
        window = max(5, int(self.cfg.early_stop_window))
        check_every = max(1, int(self.cfg.early_stop_check_every))
        patience = max(1, int(self.cfg.early_stop_patience))
        actor_thr = float(self.cfg.early_stop_actor_slope_threshold)
        critic_thr = float(self.cfg.early_stop_critic_slope_threshold)
        log_every_n_updates = 10

        for update in range(1, self.cfg.updates + 1):
            log_this_update = (update % log_every_n_updates == 0) or (update == self.cfg.updates)
            t_rollout = time.perf_counter()
            rollout = self.env.collect_rollout(
                policy=self.policy,
                value_fn=(self.value_fn if bool(self.cfg.use_critic) else None),
                batch_size=self.cfg.rollout_size,
            )
            if not bool(self.cfg.use_critic):
                reward_mean_batch = float(np.mean(rollout.rewards))
                momentum = float(np.clip(float(self.cfg.reward_baseline_momentum), 0.0, 1.0))
                if self.moving_reward_baseline is None:
                    self.moving_reward_baseline = reward_mean_batch
                else:
                    self.moving_reward_baseline = (
                        momentum * float(self.moving_reward_baseline)
                        + (1.0 - momentum) * reward_mean_batch
                    )
                baseline_val = float(self.moving_reward_baseline)
                rollout.values = np.full_like(rollout.rewards, baseline_val, dtype=np.float32)
                rollout.advantages = (rollout.returns - rollout.values).astype(np.float32)
            rollout_elapsed = time.perf_counter() - t_rollout

            t_update_inner = time.perf_counter()
            actor_loss, critic_loss, entropy = self._ppo_update(
                rollout,
                update_idx=update,
                total_updates=self.cfg.updates,
                log_update_progress=log_this_update,
            )
            inner_elapsed = time.perf_counter() - t_update_inner
            actor_loss_trace.append(float(actor_loss))
            critic_loss_trace.append(float(critic_loss))
            rollout_reward_mean = float(np.mean(rollout.rewards))

            row = {
                "update": update,
                "rollout_reward_mean": rollout_reward_mean,
                "actor_loss": actor_loss,
                "critic_loss": critic_loss,
                "entropy": entropy,
            }

            run_eval = update == 1 or update % self.cfg.eval_every == 0 or update == self.cfg.updates
            if run_eval:
                train_det = self.evaluate_split(split="train", mode="deterministic")
                test_det = self.evaluate_split(split="test", mode="deterministic")
                row.update(
                    {
                        "train_reward_det": train_det["reward_mean"],
                        "test_reward_det": test_det["reward_mean"],
                        "train_acc_det": train_det["decision_accuracy"],
                        "test_acc_det": test_det["decision_accuracy"],
                        "test_cancel_rate_det": test_det["cancel_rate"],
                        "test_cancel_success_count_det": test_det["cancel_success_count"],
                        "test_cancel_needed_count_det": test_det["cancel_needed_count"],
                        "test_cancel_bad_rate_det": test_det["cancel_needed_rate"],
                    }
                )
                if log_this_update:
                    print(
                        f"[PPO] update={update:4d}/{self.cfg.updates} "
                        f"batch={self.cfg.rollout_size} "
                        f"rollout_t={rollout_elapsed:.1f}s inner_t={inner_elapsed:.1f}s "
                        f"rollout_reward={rollout_reward_mean:+.4f} "
                        f"actor_loss={actor_loss:+.4f} critic_loss={critic_loss:+.4f} "
                        f"test_reward_det={test_det['reward_mean']:+.4f} "
                        f"test_acc_det={test_det['decision_accuracy']:.4f} "
                        f"test_cancel_rate={test_det['cancel_rate']:.4f} "
                        f"test_cancel_success={test_det['cancel_success_count']:.0f} "
                        f"test_bad_cancel={test_det['cancel_needed_count']:.0f}"
                    )
            elif log_this_update:
                print(
                    f"[PPO] update={update:4d}/{self.cfg.updates} "
                    f"batch={self.cfg.rollout_size} "
                    f"rollout_t={rollout_elapsed:.1f}s inner_t={inner_elapsed:.1f}s "
                    f"rollout_reward={rollout_reward_mean:+.4f} "
                    f"actor_loss={actor_loss:+.4f} critic_loss={critic_loss:+.4f}"
                )
            self.history.append(row)
            executed_updates = update

            if bool(self.cfg.early_stop_enabled) and update >= warmup and update % check_every == 0:
                if len(actor_loss_trace) >= window and len(critic_loss_trace) >= window:
                    actor_slope = _linear_slope(np.asarray(actor_loss_trace[-window:], dtype=np.float32))
                    critic_slope = _linear_slope(np.asarray(critic_loss_trace[-window:], dtype=np.float32))
                    actor_flat = abs(actor_slope) <= actor_thr
                    critic_flat = abs(critic_slope) <= critic_thr
                    plateau_counter = (plateau_counter + 1) if (actor_flat and critic_flat) else 0
                    self.training_status["last_actor_slope"] = float(actor_slope)
                    self.training_status["last_critic_slope"] = float(critic_slope)
                    self.training_status["plateau_counter"] = int(plateau_counter)
                    if log_this_update:
                        print(
                            f"[PPO] convergence_check update={update}/{self.cfg.updates} "
                            f"actor_slope={actor_slope:+.6f} critic_slope={critic_slope:+.6f} "
                            f"plateau={plateau_counter}/{patience}"
                        )
                    if plateau_counter >= patience:
                        stopped_early = True
                        stop_reason = (
                            f"converged_loss_slopes(window={window},actor_thr={actor_thr},critic_thr={critic_thr})"
                        )
                        print(f"[PPO] Early stop triggered at update={update}: {stop_reason}")
                        break

        elapsed = time.perf_counter() - t0
        print(f"[PPO] Training done in {_format_minutes_seconds(elapsed)}")
        self.training_status["executed_updates"] = int(executed_updates)
        self.training_status["stopped_early"] = bool(stopped_early)
        self.training_status["stop_reason"] = str(stop_reason)
        self.training_status["elapsed_s"] = float(elapsed)
        print(
            f"[PPO] Completion status: executed_updates={executed_updates}/{self.cfg.updates} "
            f"stopped_early={stopped_early} reason={stop_reason}"
        )

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

        self.policy.save_weights(str(out / "policy.weights.h5"))
        self.value_fn.save_weights(str(out / "value.weights.h5"))
        save_bundle_metadata(out / "dataset_metadata.json", self.data)
        save_json(
            out / "run_config.json",
            {
                "ppo_config": asdict(self.cfg),
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
    p = argparse.ArgumentParser(description="Train PPO agent for truck cancellation.")

    p.add_argument(
        "--config",
        type=str,
        default=default_config if default_config is not None else str(default_config_path()),
        help="Path to config_ppo.json",
    )
    p.add_argument("--stage", type=str, default=default_stage, help="Optional stage name inside config_ppo.json")
    p.add_argument("--trial-index", type=int, default=default_trial_index, help="Optional trial index in stage grid")
    p.add_argument("--list-trials", action="store_true", help="Print stage trials from config and exit")

    p.add_argument("--train-path", type=str, default=str(train_default), help="Path to df_per_dest_train.pkl")
    p.add_argument("--test-path", type=str, default=str(test_default), help="Path to df_per_dest_test.pkl")
    p.add_argument("--output-dir", type=str, default=None, help="Directory to store PPO artifacts")
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
        help="Named run folder inside --output-root (example: piperun_1)",
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
    p.add_argument("--ppo-epochs", type=int, default=8)
    p.add_argument("--minibatch-size", type=int, default=512)
    p.add_argument("--clip-ratio", type=float, default=0.2)
    p.add_argument("--actor-lr", type=float, default=3e-4)
    p.add_argument("--critic-lr", type=float, default=1e-3)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--value-coef", type=float, default=1.0)
    p.add_argument("--grad-clip-norm", type=float, default=0.5)
    p.add_argument("--gamma", type=float, default=1.0, help="Discount factor for trajectory returns")
    p.add_argument("--use-critic", dest="use_critic", action="store_true", help="Use learned critic baseline")
    p.add_argument(
        "--no-critic",
        dest="use_critic",
        action="store_false",
        help="Disable critic and use moving reward-mean baseline",
    )
    p.add_argument(
        "--reward-baseline-momentum",
        type=float,
        default=0.95,
        help="EMA momentum for moving reward baseline when critic is disabled",
    )
    p.set_defaults(use_critic=True)
    p.add_argument("--hidden-sizes", type=str, default="128,128")
    p.add_argument("--no-adv-normalize", action="store_true")
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--early-stop-enabled", action="store_true", help="Enable convergence-based early stopping")
    p.add_argument("--early-stop-warmup", type=int, default=100, help="Min updates before early-stop checks")
    p.add_argument("--early-stop-window", type=int, default=40, help="Rolling window size used for loss slopes")
    p.add_argument("--early-stop-check-every", type=int, default=10, help="Check convergence every N updates")
    p.add_argument("--early-stop-patience", type=int, default=3, help="Consecutive flat-slope checks before stop")
    p.add_argument(
        "--early-stop-actor-slope-threshold",
        type=float,
        default=1e-4,
        help="Absolute actor-loss slope threshold for convergence",
    )
    p.add_argument(
        "--early-stop-critic-slope-threshold",
        type=float,
        default=5e-4,
        help="Absolute critic-loss slope threshold for convergence",
    )

    p.add_argument("--decision-threshold", type=float, default=0.5)
    p.add_argument("--stochastic-min-prob", type=float, default=0.05)
    p.add_argument("--stochastic-max-prob", type=float, default=0.95)

    p.add_argument("--reward-keep-needed", type=float, default=1.0)
    p.add_argument("--reward-cancel-not-needed", type=float, default=1.0)
    p.add_argument("--reward-cancel-needed", type=float, default=-10.0)
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
        config_doc = load_ppo_config(config_path)
        if pre_args.list_trials:
            stage_name, trials, tag_keys = list_stage_trials(config_doc, stage=pre_args.stage)
            print(f"[PPO] config={config_path}")
            if stage_name is None:
                print("[PPO] No stages configured.")
            else:
                print(f"[PPO] stage={stage_name} trials={len(trials)}")
                if tag_keys:
                    print(f"[PPO] tag_keys={tag_keys}")
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

    ppo_cfg = PPOConfig(
        seed=int(args.seed),
        updates=int(args.updates),
        rollout_size=int(args.rollout_size),
        ppo_epochs=int(args.ppo_epochs),
        minibatch_size=int(args.minibatch_size),
        clip_ratio=float(args.clip_ratio),
        actor_lr=float(args.actor_lr),
        critic_lr=float(args.critic_lr),
        entropy_coef=float(args.entropy_coef),
        value_coef=float(args.value_coef),
        grad_clip_norm=float(args.grad_clip_norm),
        gamma=float(args.gamma),
        use_critic=bool(args.use_critic),
        reward_baseline_momentum=float(args.reward_baseline_momentum),
        hidden_sizes=hidden_sizes,
        normalize_advantages=(not args.no_adv_normalize),
        eval_every=int(args.eval_every),
        decision_threshold=float(args.decision_threshold),
        stochastic_min_prob=float(args.stochastic_min_prob),
        stochastic_max_prob=float(args.stochastic_max_prob),
        early_stop_enabled=bool(args.early_stop_enabled),
        early_stop_warmup=int(args.early_stop_warmup),
        early_stop_window=int(args.early_stop_window),
        early_stop_check_every=int(args.early_stop_check_every),
        early_stop_patience=int(args.early_stop_patience),
        early_stop_actor_slope_threshold=float(args.early_stop_actor_slope_threshold),
        early_stop_critic_slope_threshold=float(args.early_stop_critic_slope_threshold),
    )

    trainer = PPOTruckCancellationOptimiser(data=data, reward_config=reward_cfg, cfg=ppo_cfg)
    final_metrics = trainer.train()

    if args.output_dir is None:
        if args.run_name:
            out_dir = Path(args.output_root).expanduser() / str(args.run_name)
        else:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            out_dir = Path(args.output_root).expanduser() / f"ppo_{timestamp}"
    else:
        out_dir = Path(args.output_dir).expanduser()
    saved_dir = trainer.save(out_dir, final_metrics, extra_config=config_meta)

    print("[PPO] Final metrics:")
    for key, val in final_metrics.items():
        print(
            f"  - {key}: reward={val['reward_mean']:+.4f}, acc={val['decision_accuracy']:.4f}, "
            f"cancel_rate={val['cancel_rate']:.4f}, "
            f"cancel_success={val['cancel_success_count']:.0f}, "
            f"cancel_needed={val['cancel_needed_count']:.0f}"
        )
    print(f"[PPO] Artifacts saved to: {saved_dir}")
    print(f"[PPO] Training status: {saved_dir / 'training_status.json'}")
    print(f"[PPO] Plot manifest: {saved_dir / 'plots_manifest.json'}")


if __name__ == "__main__":
    main()
