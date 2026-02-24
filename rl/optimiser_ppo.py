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
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
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


@dataclass
class PPOConfig:
    """Hyperparameters for one-step PPO training."""

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
    hidden_sizes: tuple[int, ...] = (128, 128)
    normalize_advantages: bool = True
    eval_every: int = 10

    # Decision rules used for evaluation/deployment.
    decision_threshold: float = 0.5
    stochastic_min_prob: float = 0.05
    stochastic_max_prob: float = 0.95


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

        self.env = OneStepCancellationEnv(
            states=self.data.x_train,
            labels_needed=self.data.y_train,
            reward_config=self.reward_cfg,
            hours=self.data.train_hours,
            min_hour=self.data.min_hour,
            max_hour=self.data.max_hour,
            rng=self.rng,
        )

        self.history: list[dict] = []

    def _ppo_update(self, rollout) -> tuple[float, float, float]:
        idx_all = np.arange(rollout.size, dtype=np.int32)
        actor_losses: list[float] = []
        critic_losses: list[float] = []
        entropies: list[float] = []

        adv = rollout.advantages.astype(np.float32, copy=True)
        if self.cfg.normalize_advantages:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        for _ in range(self.cfg.ppo_epochs):
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

                actor_losses.append(float(actor_loss.numpy()))
                critic_losses.append(float(critic_loss.numpy()))
                entropies.append(float(entropy.numpy()))

        return (
            float(np.mean(actor_losses)) if actor_losses else 0.0,
            float(np.mean(critic_losses)) if critic_losses else 0.0,
            float(np.mean(entropies)) if entropies else 0.0,
        )

    def evaluate_split(
        self,
        split: str = "test",
        mode: str = "deterministic",
    ) -> dict[str, float]:
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

        cancel_prob = predict_cancel_probability(self.policy, x)
        actions = choose_actions(
            cancel_prob=cancel_prob,
            mode=mode,
            threshold=float(self.cfg.decision_threshold),
            rng=self.rng,
            min_prob=float(self.cfg.stochastic_min_prob),
            max_prob=float(self.cfg.stochastic_max_prob),
        )
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

    def train(self) -> dict[str, dict[str, float]]:
        print(
            f"[PPO] Start training: state_dim={self.data.state_dim}, "
            f"n_train={self.data.x_train.shape[0]}, n_test={self.data.x_test.shape[0]}"
        )

        t0 = time.perf_counter()
        for update in range(1, self.cfg.updates + 1):
            rollout = self.env.collect_rollout(
                policy=self.policy,
                value_fn=self.value_fn,
                batch_size=self.cfg.rollout_size,
            )
            actor_loss, critic_loss, entropy = self._ppo_update(rollout)
            rollout_reward_mean = float(np.mean(rollout.rewards))

            row = {
                "update": update,
                "rollout_reward_mean": rollout_reward_mean,
                "actor_loss": actor_loss,
                "critic_loss": critic_loss,
                "entropy": entropy,
            }

            if update == 1 or update % self.cfg.eval_every == 0 or update == self.cfg.updates:
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
                print(
                    f"[PPO] update={update:4d}/{self.cfg.updates} "
                    f"rollout_reward={rollout_reward_mean:+.4f} "
                    f"actor_loss={actor_loss:+.4f} critic_loss={critic_loss:+.4f} "
                    f"test_reward_det={test_det['reward_mean']:+.4f} "
                    f"test_acc_det={test_det['decision_accuracy']:.4f} "
                    f"test_cancel_rate={test_det['cancel_rate']:.4f} "
                    f"test_cancel_success={test_det['cancel_success_count']:.0f} "
                    f"test_bad_cancel={test_det['cancel_needed_count']:.0f}"
                )
            self.history.append(row)

        elapsed = time.perf_counter() - t0
        print(f"[PPO] Training done in {elapsed:.2f}s")

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
        default="/disk/lhcb_data/maander/output_truck_cancellation",
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
        default="fill_threshold",
        help="How to derive needed-truck label: fill_threshold|dataset_label",
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
    p.add_argument("--hidden-sizes", type=str, default="128,128")
    p.add_argument("--no-adv-normalize", action="store_true")
    p.add_argument("--eval-every", type=int, default=10)

    p.add_argument("--decision-threshold", type=float, default=0.5)
    p.add_argument("--stochastic-min-prob", type=float, default=0.05)
    p.add_argument("--stochastic-max-prob", type=float, default=0.95)

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
        hidden_sizes=hidden_sizes,
        normalize_advantages=(not args.no_adv_normalize),
        eval_every=int(args.eval_every),
        decision_threshold=float(args.decision_threshold),
        stochastic_min_prob=float(args.stochastic_min_prob),
        stochastic_max_prob=float(args.stochastic_max_prob),
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


if __name__ == "__main__":
    main()
