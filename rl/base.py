#!/usr/bin/env python3
"""Shared utilities for truck-cancellation RL optimisers.

This module centralizes functionality reused by multiple optimisers:
- data loading and feature construction for toy_sim datasets
- reward shaping for cancellation decisions
- one-step rollout building for offline RL-style training
- common neural modules, including Rainbow-oriented components
"""

from __future__ import annotations

import importlib
import json
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model, layers


DEFAULT_NUMERIC_FEATURES: tuple[str, ...] = (
    "hour",
    "min",
    "max",
    "mean",
    "std",
    "delta",
    "day",
    "season",
    "hist_avg_vol_tot",
    "hist_std_vol_tot",
    "n_exp_trucks",
    "frac_last_truck_needed",
    "hist_avg_vol",
    "hist_std_vol",
)


@dataclass
class FeatureConfig:
    """Feature set definition for state construction."""

    numeric_features: tuple[str, ...] = DEFAULT_NUMERIC_FEATURES
    include_center_one_hot: bool = True
    include_dest_one_hot: bool = True
    normalize_numeric: bool = True
    zscore_clip: float = 8.0


@dataclass
class LabelConfig:
    """How to derive the "truck needed" target used for rewards/training."""

    source: str = "dataset_label"
    needed_fill_threshold: float = 0.2
    n_parcels_per_truck: int = 100


@dataclass
class RewardConfig:
    """Reward matrix for action/label outcomes.

    Action mapping:
    - `0` -> keep last truck
    - `1` -> cancel last truck

    Label mapping:
    - `1` -> truck needed
    - `0` -> truck not needed
    """

    keep_when_needed: float = 1.0
    cancel_when_not_needed: float = 1.0
    cancel_when_needed: float = -5.0
    keep_when_not_needed: float = -1.0
    early_cancel_bonus: float = 0.5
    early_cancel_penalty: float = 0.5


@dataclass
class Normalizer:
    """Simple z-score normalizer for numeric features."""

    mean: np.ndarray | None = None
    std: np.ndarray | None = None
    clip: float = 8.0

    def fit(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float32)
        self.mean = x.mean(axis=0).astype(np.float32)
        self.std = x.std(axis=0).astype(np.float32)
        self.std = np.where(self.std < 1e-6, 1.0, self.std).astype(np.float32)

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if self.mean is None or self.std is None:
            return x
        out = (x - self.mean) / self.std
        if self.clip > 0.0:
            out = np.clip(out, -self.clip, self.clip)
        return out.astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": None if self.mean is None else self.mean.tolist(),
            "std": None if self.std is None else self.std.tolist(),
            "clip": float(self.clip),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Normalizer":
        mean = payload.get("mean")
        std = payload.get("std")
        return cls(
            mean=None if mean is None else np.asarray(mean, dtype=np.float32),
            std=None if std is None else np.asarray(std, dtype=np.float32),
            clip=float(payload.get("clip", 8.0)),
        )


@dataclass
class DataBundle:
    """Packaged train/test data and metadata used by optimisers."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    feature_config: FeatureConfig
    label_config: LabelConfig
    normalizer: Normalizer
    train_hours: np.ndarray | None = None
    test_hours: np.ndarray | None = None
    min_hour: int = 0
    max_hour: int = 1
    fill_ratio_train: np.ndarray | None = None
    fill_ratio_test: np.ndarray | None = None
    center_vocab: list[str] = field(default_factory=list)
    dest_vocab: list[str] = field(default_factory=list)
    train_df: pd.DataFrame | None = None
    test_df: pd.DataFrame | None = None
    train_episode_ids: np.ndarray | None = None
    test_episode_ids: np.ndarray | None = None

    @property
    def state_dim(self) -> int:
        return int(self.x_train.shape[1])


@dataclass
class RolloutBatch:
    """One-step on-policy rollout data used by PPO-style updates."""

    states: np.ndarray
    labels: np.ndarray
    actions: np.ndarray
    old_log_probs: np.ndarray
    rewards: np.ndarray
    hours: np.ndarray
    values: np.ndarray
    returns: np.ndarray
    advantages: np.ndarray

    @property
    def size(self) -> int:
        return int(self.states.shape[0])


@dataclass
class ReplaySample:
    """Replay minibatch with optional PER metadata."""

    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_obs: np.ndarray
    dones: np.ndarray
    discounts: np.ndarray
    is_weights: np.ndarray
    indices: np.ndarray | None = None


def seed_everything(seed: int | None) -> np.random.Generator:
    """Seed python/numpy/tensorflow and return a numpy Generator."""
    s = 0 if seed is None else int(seed)
    random.seed(s)
    np.random.seed(s)
    try:
        tf.random.set_seed(s)
    except Exception:
        pass
    return np.random.default_rng(s)


def default_dataset_paths(base_dir: Path | None = None) -> tuple[Path, Path]:
    """Return default train/test pickle paths under toy_sim/example_data."""
    base = Path(base_dir) if base_dir is not None else Path(__file__).resolve().parents[1]
    data_dir = base / "toy_sim" / "example_data"
    return data_dir / "df_per_dest_train.pkl", data_dir / "df_per_dest_test.pkl"


def _load_pickle_df(path: str | Path) -> pd.DataFrame:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p}")
    try:
        df = pd.read_pickle(p)
    except ModuleNotFoundError as exc:
        if "numpy._core" not in str(exc):
            raise
        _install_numpy_pickle_compat_aliases()
        df = pd.read_pickle(p)
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Loaded object from {p} is not a pandas DataFrame")
    return df


def _install_numpy_pickle_compat_aliases() -> None:
    """Alias NumPy 2.x pickle module paths when loading with NumPy 1.x."""
    alias_map = {
        "numpy._core": "numpy.core",
        "numpy._core.numeric": "numpy.core.numeric",
        "numpy._core.multiarray": "numpy.core.multiarray",
        "numpy._core.umath": "numpy.core.umath",
        "numpy._core._multiarray_umath": "numpy.core._multiarray_umath",
    }
    for alias_name, target_name in alias_map.items():
        if alias_name in sys.modules:
            continue
        try:
            sys.modules[alias_name] = importlib.import_module(target_name)
        except Exception:
            continue


def _require_columns(df: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _build_ohe(values: Sequence[str], vocab: Sequence[str]) -> np.ndarray:
    vocab_idx = {val: i for i, val in enumerate(vocab)}
    arr = np.zeros((len(values), len(vocab)), dtype=np.float32)
    for row, value in enumerate(values):
        idx = vocab_idx.get(value)
        if idx is not None:
            arr[row, idx] = 1.0
    return arr


def build_state_matrices(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: FeatureConfig,
) -> tuple[np.ndarray, np.ndarray, list[str], Normalizer, list[str], list[str]]:
    """Convert dataframes into state matrices for RL training."""
    required = list(cfg.numeric_features) + ["last_truck_needed"]
    _require_columns(train_df, required, "train_df")
    _require_columns(test_df, required, "test_df")

    feature_names: list[str] = []
    x_train_parts: list[np.ndarray] = []
    x_test_parts: list[np.ndarray] = []

    x_train_num = train_df.loc[:, cfg.numeric_features].to_numpy(dtype=np.float32)
    x_test_num = test_df.loc[:, cfg.numeric_features].to_numpy(dtype=np.float32)
    normalizer = Normalizer(clip=float(cfg.zscore_clip))
    if cfg.normalize_numeric:
        normalizer.fit(x_train_num)
        x_train_num = normalizer.transform(x_train_num)
        x_test_num = normalizer.transform(x_test_num)
    x_train_parts.append(x_train_num)
    x_test_parts.append(x_test_num)
    feature_names.extend(list(cfg.numeric_features))

    center_vocab: list[str] = []
    dest_vocab: list[str] = []

    if cfg.include_center_one_hot:
        if "center" not in train_df.columns or "center" not in test_df.columns:
            raise ValueError("FeatureConfig requests center one-hot, but 'center' column is missing.")
        center_vocab = sorted(train_df["center"].astype(str).unique().tolist())
        train_center = _build_ohe(train_df["center"].astype(str).tolist(), center_vocab)
        test_center = _build_ohe(test_df["center"].astype(str).tolist(), center_vocab)
        x_train_parts.append(train_center)
        x_test_parts.append(test_center)
        feature_names.extend([f"center={x}" for x in center_vocab])

    if cfg.include_dest_one_hot:
        if "dest" not in train_df.columns or "dest" not in test_df.columns:
            raise ValueError("FeatureConfig requests destination one-hot, but 'dest' column is missing.")
        dest_vocab = sorted(train_df["dest"].astype(str).unique().tolist())
        train_dest = _build_ohe(train_df["dest"].astype(str).tolist(), dest_vocab)
        test_dest = _build_ohe(test_df["dest"].astype(str).tolist(), dest_vocab)
        x_train_parts.append(train_dest)
        x_test_parts.append(test_dest)
        feature_names.extend([f"dest={x}" for x in dest_vocab])

    x_train = np.concatenate(x_train_parts, axis=1).astype(np.float32)
    x_test = np.concatenate(x_test_parts, axis=1).astype(np.float32)
    return x_train, x_test, feature_names, normalizer, center_vocab, dest_vocab


def estimate_last_truck_fill_ratio(df: pd.DataFrame, n_parcels_per_truck: int = 100) -> np.ndarray:
    """Estimate last-truck fill ratio from historical route volume features.

    Required columns:
    - `hist_avg_vol`
    - `n_exp_trucks`
    """
    _require_columns(df, ["hist_avg_vol", "n_exp_trucks"], "dataframe")
    cap = float(max(1, int(n_parcels_per_truck)))
    hist_avg = df["hist_avg_vol"].to_numpy(dtype=np.float32)
    n_exp = np.maximum(df["n_exp_trucks"].to_numpy(dtype=np.float32), 1.0)
    last_truck_letters = hist_avg - (n_exp - 1.0) * cap
    fill_ratio = last_truck_letters / cap
    return np.clip(fill_ratio, 0.0, 2.0).astype(np.float32)


def derive_needed_label(df: pd.DataFrame, cfg: LabelConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return (needed_label, fill_ratio) based on selected label source."""
    source = str(cfg.source).strip().lower()
    fill_ratio = estimate_last_truck_fill_ratio(df, n_parcels_per_truck=cfg.n_parcels_per_truck)

    if source in {"dataset", "dataset_label", "sim_label", "raw_label"}:
        _require_columns(df, ["last_truck_needed"], "dataframe")
        y = df["last_truck_needed"].to_numpy(dtype=np.float32)
    elif source in {"fill", "fill_ratio", "fill_threshold"}:
        thr = float(cfg.needed_fill_threshold)
        y = (fill_ratio >= thr).astype(np.float32)
    else:
        raise ValueError(
            f"Unknown LabelConfig.source={cfg.source!r}. "
            "Use dataset_label|sim_label or fill_threshold."
        )

    y = (y >= 0.5).astype(np.float32)
    return y, fill_ratio


def derive_episode_ids(df: pd.DataFrame | None) -> np.ndarray | None:
    """Build per-row episode identifiers for trajectory-style training.

    Preferred source columns (if present): `sample_id`, `episode_id`, `day_id`.
    Fallback (for existing macro datasets): align rows across hours using
    row-order rank within each (center, dest, hour) group.
    """
    if df is None or len(df) == 0:
        return None

    for col in ("sample_id", "episode_id", "day_id"):
        if col in df.columns:
            base = df[col].astype(str).to_numpy()
            if "center" in df.columns and "dest" in df.columns:
                keys = (
                    df["center"].astype(str).to_numpy()
                    + "|"
                    + df["dest"].astype(str).to_numpy()
                    + "|"
                    + base
                )
            else:
                keys = base
            codes, _uniques = pd.factorize(keys, sort=False)
            return np.asarray(codes, dtype=np.int32)

    required = {"center", "dest", "hour"}
    if required.issubset(set(df.columns)):
        temp = pd.DataFrame(
            {
                "center": df["center"].astype(str).to_numpy(),
                "dest": df["dest"].astype(str).to_numpy(),
                "hour": df["hour"].to_numpy(dtype=np.int32),
            }
        )
        temp["_seq"] = temp.groupby(["center", "dest", "hour"], sort=False).cumcount()
        keys = (
            temp["center"].to_numpy()
            + "|"
            + temp["dest"].to_numpy()
            + "|"
            + temp["_seq"].astype(str).to_numpy()
        )
        codes, _uniques = pd.factorize(keys, sort=False)
        return np.asarray(codes, dtype=np.int32)

    # Last-resort fallback: each row is its own "episode".
    return np.arange(len(df), dtype=np.int32)


def load_data_bundle(
    train_path: str | Path,
    test_path: str | Path,
    feature_config: FeatureConfig | None = None,
    label_config: LabelConfig | None = None,
) -> DataBundle:
    """Load train/test pickles and build numeric state tensors."""
    cfg = feature_config or FeatureConfig()
    lcfg = label_config or LabelConfig()
    train_df = _load_pickle_df(train_path)
    test_df = _load_pickle_df(test_path)

    x_train, x_test, feature_names, norm, center_vocab, dest_vocab = build_state_matrices(train_df, test_df, cfg)
    y_train, fill_ratio_train = derive_needed_label(train_df, lcfg)
    y_test, fill_ratio_test = derive_needed_label(test_df, lcfg)
    train_episode_ids = derive_episode_ids(train_df)
    test_episode_ids = derive_episode_ids(test_df)
    train_hours = (
        train_df["hour"].to_numpy(dtype=np.float32)
        if "hour" in train_df.columns
        else np.zeros((x_train.shape[0],), dtype=np.float32)
    )
    test_hours = (
        test_df["hour"].to_numpy(dtype=np.float32)
        if "hour" in test_df.columns
        else np.zeros((x_test.shape[0],), dtype=np.float32)
    )
    max_hour = int(max(np.max(train_hours), np.max(test_hours), 1.0))
    min_hour = int(min(np.min(train_hours), np.min(test_hours)))

    return DataBundle(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        feature_names=feature_names,
        feature_config=cfg,
        label_config=lcfg,
        normalizer=norm,
        train_hours=train_hours,
        test_hours=test_hours,
        min_hour=min_hour,
        max_hour=max_hour,
        fill_ratio_train=fill_ratio_train,
        fill_ratio_test=fill_ratio_test,
        center_vocab=center_vocab,
        dest_vocab=dest_vocab,
        train_df=train_df,
        test_df=test_df,
        train_episode_ids=train_episode_ids,
        test_episode_ids=test_episode_ids,
    )


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid for numpy arrays."""
    x = np.asarray(x, dtype=np.float32)
    out = np.empty_like(x, dtype=np.float32)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[neg])
    out[neg] = exp_x / (1.0 + exp_x)
    return out


def _softplus_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def bernoulli_log_prob_np(actions: np.ndarray, logits: np.ndarray) -> np.ndarray:
    """Log probability log pi(a|s) for Bernoulli policy parameterized by logits."""
    a = np.asarray(actions, dtype=np.float32)
    z = np.asarray(logits, dtype=np.float32)
    return a * (-_softplus_np(-z)) + (1.0 - a) * (-_softplus_np(z))


def bernoulli_log_prob_from_logits(logits: tf.Tensor, actions: tf.Tensor) -> tf.Tensor:
    """TensorFlow equivalent of Bernoulli log probability."""
    a = tf.cast(actions, tf.float32)
    z = tf.cast(logits, tf.float32)
    return a * (-tf.nn.softplus(-z)) + (1.0 - a) * (-tf.nn.softplus(z))


def bernoulli_entropy_from_logits(logits: tf.Tensor) -> tf.Tensor:
    """Per-sample entropy of a Bernoulli distribution parameterized by logits."""
    z = tf.cast(logits, tf.float32)
    p = tf.math.sigmoid(z)
    p = tf.clip_by_value(p, 1e-7, 1.0 - 1e-7)
    return -(p * tf.math.log(p) + (1.0 - p) * tf.math.log(1.0 - p))


def choose_actions(
    cancel_prob: np.ndarray,
    mode: str = "deterministic",
    threshold: float = 0.5,
    rng: np.random.Generator | None = None,
    min_prob: float = 0.0,
    max_prob: float = 1.0,
) -> np.ndarray:
    """Convert cancellation probabilities to binary keep/cancel decisions."""
    p = np.asarray(cancel_prob, dtype=np.float32)
    mode_norm = str(mode).strip().lower()
    if mode_norm == "deterministic":
        return (p >= float(threshold)).astype(np.int32)
    if mode_norm == "stochastic":
        gen = rng if rng is not None else np.random.default_rng()
        lo = float(max(0.0, min(1.0, min_prob)))
        hi = float(max(0.0, min(1.0, max_prob)))
        if hi < lo:
            lo, hi = hi, lo
        p = np.clip(p, lo, hi)
        return gen.binomial(1, p=p).astype(np.int32)
    raise ValueError(f"Unknown action mode: {mode!r}. Expected deterministic|stochastic.")


def compute_rewards(
    actions: np.ndarray,
    labels_needed: np.ndarray,
    cfg: RewardConfig,
    hours: np.ndarray | None = None,
    min_hour: int | None = None,
    max_hour: int | None = None,
) -> np.ndarray:
    """Compute scalar reward per one-step episode.

    Includes optional time-weighting so earlier cancellation can have larger magnitude.
    """
    action_cancel = np.asarray(actions, dtype=np.int32) == 1
    needed = np.asarray(labels_needed, dtype=np.float32) >= 0.5

    rewards = np.empty_like(np.asarray(labels_needed, dtype=np.float32), dtype=np.float32)
    rewards[(~action_cancel) & needed] = float(cfg.keep_when_needed)
    rewards[action_cancel & (~needed)] = float(cfg.cancel_when_not_needed)
    rewards[action_cancel & needed] = float(cfg.cancel_when_needed)
    rewards[(~action_cancel) & (~needed)] = float(cfg.keep_when_not_needed)

    if hours is not None:
        h = np.asarray(hours, dtype=np.float32)
        if h.shape[0] != rewards.shape[0]:
            raise ValueError(f"hours length mismatch: got {h.shape[0]}, expected {rewards.shape[0]}")
        if min_hour is not None:
            hmin = float(min(float(min_hour), np.min(h)))
        else:
            hmin = float(np.min(h))
        if max_hour is not None:
            hmax = float(max(float(max_hour), np.max(h)))
        else:
            hmax = float(np.max(h))
        if hmax <= hmin + 1e-8:
            early_factor = np.ones_like(h, dtype=np.float32)
        else:
            # Earliest timestep gets 1.0, latest gets 0.0.
            early_factor = np.clip((hmax - h) / (hmax - hmin), 0.0, 1.0).astype(np.float32)

        success_cancel = action_cancel & (~needed)
        failed_cancel = action_cancel & needed
        if float(cfg.early_cancel_bonus) != 0.0:
            rewards[success_cancel] += float(cfg.early_cancel_bonus) * early_factor[success_cancel]
        if float(cfg.early_cancel_penalty) != 0.0:
            rewards[failed_cancel] -= float(cfg.early_cancel_penalty) * early_factor[failed_cancel]
    return rewards


def compute_decision_metrics(actions: np.ndarray, labels_needed: np.ndarray, rewards: np.ndarray) -> dict[str, float]:
    """Summarize outcomes for keep/cancel decisions."""
    a = np.asarray(actions, dtype=np.int32)
    y = (np.asarray(labels_needed, dtype=np.float32) >= 0.5).astype(np.int32)
    r = np.asarray(rewards, dtype=np.float32)
    n = max(1, len(a))

    keep_needed = int(np.sum((a == 0) & (y == 1)))
    cancel_not_needed = int(np.sum((a == 1) & (y == 0)))
    cancel_needed = int(np.sum((a == 1) & (y == 1)))
    keep_not_needed = int(np.sum((a == 0) & (y == 0)))
    cancel_count = int(np.sum(a == 1))

    cancel_precision = cancel_not_needed / max(1, cancel_not_needed + cancel_needed)
    cancel_recall = cancel_not_needed / max(1, cancel_not_needed + keep_not_needed)
    f1_denom = cancel_precision + cancel_recall
    cancel_f1 = (
        (2.0 * cancel_precision * cancel_recall / f1_denom)
        if f1_denom > 0.0
        else 0.0
    )

    beta = 0.5
    beta_sq = beta * beta
    f_beta_denom = beta_sq * cancel_precision + cancel_recall
    cancel_fbeta_beta05 = (
        ((1.0 + beta_sq) * cancel_precision * cancel_recall / f_beta_denom)
        if f_beta_denom > 0.0
        else 0.0
    )

    return {
        "n": float(n),
        "reward_mean": float(np.mean(r)),
        "reward_std": float(np.std(r)),
        "decision_accuracy": float((keep_needed + cancel_not_needed) / n),
        "cancel_rate": float(np.mean(a == 1)),
        "needed_rate": float(np.mean(y == 1)),
        "cancel_count": float(cancel_count),
        "cancel_success_count": float(cancel_not_needed),
        "cancel_needed_count": float(cancel_needed),
        "keep_needed_rate": float(keep_needed / n),
        "cancel_not_needed_rate": float(cancel_not_needed / n),
        "cancel_needed_rate": float(cancel_needed / n),
        "keep_not_needed_rate": float(keep_not_needed / n),
        "cancel_precision": float(cancel_precision),
        "cancel_recall": float(cancel_recall),
        "cancel_f1": float(cancel_f1),
        "cancel_fbeta": float(cancel_fbeta_beta05),
        "cancel_fbeta_beta05": float(cancel_fbeta_beta05),
        "cancel_fbeta_beta1": float(cancel_f1),
        "cancel_success_rate_among_cancellations": float(cancel_precision),
        "cancel_needed_rate_among_cancellations": float(1.0 - cancel_precision),
    }


def apply_gradients_clipped(
    optimizer: tf.keras.optimizers.Optimizer,
    loss: tf.Tensor,
    variables: Sequence[tf.Variable],
    tape: tf.GradientTape,
    grad_clip_norm: float | None = None,
) -> None:
    """Apply gradients with optional global-norm clipping."""
    grads = tape.gradient(loss, variables)
    grads_and_vars = [(g, v) for g, v in zip(grads, variables) if g is not None]
    if not grads_and_vars:
        return
    if grad_clip_norm is not None and grad_clip_norm > 0.0:
        g_list, v_list = zip(*grads_and_vars)
        g_list, _ = tf.clip_by_global_norm(g_list, float(grad_clip_norm))
        grads_and_vars = list(zip(g_list, v_list))
    optimizer.apply_gradients(grads_and_vars)


def linear_schedule(start: float, end: float, step: int, total_steps: int) -> float:
    """Linear interpolation schedule."""
    if total_steps <= 1:
        return float(end)
    alpha = max(0.0, min(1.0, float(step) / float(total_steps - 1)))
    return float(start + (end - start) * alpha)


class BinaryPolicyNetwork(Model):
    """Bernoulli policy network producing cancellation logits."""

    def __init__(self, input_dim: int, hidden_sizes: Sequence[int] = (128, 128), activation: str = "relu"):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden = [layers.Dense(int(h), activation=activation) for h in hidden_sizes]
        self.logit_head = layers.Dense(1, activation=None)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        z = tf.convert_to_tensor(x, tf.float32)
        for layer in self.hidden:
            z = layer(z, training=training)
        return self.logit_head(z, training=training)


class ValueNetwork(Model):
    """State-value baseline network."""

    def __init__(self, input_dim: int, hidden_sizes: Sequence[int] = (128, 128), activation: str = "relu"):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden = [layers.Dense(int(h), activation=activation) for h in hidden_sizes]
        self.value_head = layers.Dense(1, activation=None)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        z = tf.convert_to_tensor(x, tf.float32)
        for layer in self.hidden:
            z = layer(z, training=training)
        return self.value_head(z, training=training)


class NoisyDense(layers.Layer):
    """Factorized NoisyNet layer (for future Rainbow-like value optimisers)."""

    def __init__(
        self,
        units: int,
        sigma0: float = 0.5,
        activation: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.units = int(units)
        self.sigma0 = float(sigma0)
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        in_dim = int(input_shape[-1])
        mu_range = 1.0 / np.sqrt(float(in_dim))

        self.w_mu = self.add_weight(
            name="w_mu",
            shape=(in_dim, self.units),
            initializer=tf.keras.initializers.RandomUniform(-mu_range, mu_range),
            trainable=True,
        )
        self.w_sigma = self.add_weight(
            name="w_sigma",
            shape=(in_dim, self.units),
            initializer=tf.keras.initializers.Constant(self.sigma0 / np.sqrt(float(in_dim))),
            trainable=True,
        )
        self.b_mu = self.add_weight(
            name="b_mu",
            shape=(self.units,),
            initializer=tf.keras.initializers.RandomUniform(-mu_range, mu_range),
            trainable=True,
        )
        self.b_sigma = self.add_weight(
            name="b_sigma",
            shape=(self.units,),
            initializer=tf.keras.initializers.Constant(self.sigma0 / np.sqrt(float(in_dim))),
            trainable=True,
        )

    @staticmethod
    def _factor_noise(dim: int) -> tf.Tensor:
        x = tf.random.normal((dim,), dtype=tf.float32)
        return tf.sign(x) * tf.sqrt(tf.abs(x))

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        z = tf.convert_to_tensor(x, tf.float32)
        if training:
            eps_in = self._factor_noise(int(self.w_mu.shape[0]))
            eps_out = self._factor_noise(int(self.w_mu.shape[1]))
            eps_w = tf.einsum("i,j->ij", eps_in, eps_out)
            eps_b = eps_out
            w = self.w_mu + self.w_sigma * eps_w
            b = self.b_mu + self.b_sigma * eps_b
        else:
            w = self.w_mu
            b = self.b_mu
        out = tf.linalg.matmul(z, w) + b
        if self.activation is not None:
            out = self.activation(out)
        return out


class DuelingQNetwork(Model):
    """Common dueling architecture for IQN/QR style critics."""

    def __init__(
        self,
        input_dim: int,
        num_actions: int,
        hidden_sizes: Sequence[int] = (128, 128),
        dueling: bool = True,
        noisy: bool = False,
        activation: str = "relu",
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_actions = int(num_actions)
        self.dueling = bool(dueling)

        dense_cls = NoisyDense if noisy else layers.Dense
        self.hidden = [dense_cls(int(h), activation=activation) for h in hidden_sizes]
        if self.dueling:
            self.v_head = dense_cls(1, activation=None)
            self.adv_head = dense_cls(self.num_actions, activation=None)
        else:
            self.q_head = dense_cls(self.num_actions, activation=None)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        z = tf.convert_to_tensor(x, tf.float32)
        for layer in self.hidden:
            z = layer(z, training=training)
        if self.dueling:
            v = self.v_head(z, training=training)
            a = self.adv_head(z, training=training)
            a = a - tf.reduce_mean(a, axis=1, keepdims=True)
            return v + a
        return self.q_head(z, training=training)


def make_dense_layer(
    units: int,
    activation: str | None = "relu",
    noisy: bool = False,
    name: str | None = None,
):
    """Build a Dense/NoisyDense layer with a common signature."""
    if noisy:
        return NoisyDense(int(units), activation=activation, name=name)
    return layers.Dense(int(units), activation=activation, name=name)


def huber_loss(errors: tf.Tensor, kappa: float = 1.0) -> tf.Tensor:
    """Huber loss on arbitrary-shaped tensors."""
    err = tf.convert_to_tensor(errors, dtype=tf.float32)
    k = float(kappa)
    if k <= 0.0:
        return tf.abs(err)
    abs_err = tf.abs(err)
    return tf.where(abs_err <= k, 0.5 * tf.square(err), k * (abs_err - 0.5 * k))


def quantile_huber_loss_per_sample(
    td_errors: tf.Tensor,
    taus: tf.Tensor,
    kappa: float = 1.0,
) -> tf.Tensor:
    """Quantile Huber loss per sample for QR/IQN-style critics.

    Args:
        td_errors: Tensor shaped [B, N, N_tgt] where N=current quantiles.
        taus: Tensor shaped [B, N] or [N] corresponding to current quantiles.
        kappa: Huber threshold.
    """
    td = tf.convert_to_tensor(td_errors, dtype=tf.float32)
    tau = tf.convert_to_tensor(taus, dtype=tf.float32)
    if tau.shape.rank == 1:
        batch_size = tf.shape(td)[0]
        tau = tf.broadcast_to(tau[None, :], (batch_size, tf.shape(tau)[0]))
    if tau.shape.rank != 2:
        raise ValueError(f"taus must be rank-1 or rank-2, got shape {tau.shape}")

    indicator = tf.cast(td < 0.0, tf.float32)
    tau_expand = tau[:, :, None]
    quantile_weights = tf.abs(tau_expand - indicator)
    loss = quantile_weights * huber_loss(td, kappa=kappa)
    return tf.reduce_mean(loss, axis=[1, 2])


def reduce_per_sample_loss(loss_per_sample: tf.Tensor, is_weights: tf.Tensor | None = None) -> tf.Tensor:
    """Reduce per-sample losses with optional normalized importance weights."""
    per_sample = tf.convert_to_tensor(loss_per_sample, dtype=tf.float32)
    if is_weights is None:
        return tf.reduce_mean(per_sample)
    weights = tf.convert_to_tensor(is_weights, dtype=tf.float32)
    weights = weights / (tf.reduce_max(weights) + 1e-8)
    return tf.reduce_mean(per_sample * weights)


def soft_update_model(source: Model, target: Model, tau: float = 0.005) -> None:
    """Polyak averaging update: target <- tau*source + (1-tau)*target."""
    t = float(np.clip(float(tau), 0.0, 1.0))
    if t >= 1.0:
        target.set_weights(source.get_weights())
        return
    for src_var, tgt_var in zip(source.trainable_variables, target.trainable_variables):
        tgt_var.assign(t * src_var + (1.0 - t) * tgt_var)


def hard_update_model(source: Model, target: Model) -> None:
    """Copy model weights from source into target."""
    target.set_weights(source.get_weights())


class IQNQNetwork(Model):
    """Implicit Quantile Network for discrete actions.

    Output shape is [B, A, N] where:
    - B: batch size
    - A: number of discrete actions
    - N: number of sampled quantiles
    """

    def __init__(
        self,
        input_dim: int,
        num_actions: int,
        hidden_sizes: Sequence[int] = (128, 128),
        n_cos: int = 64,
        dueling: bool = True,
        noisy: bool = False,
        activation: str = "relu",
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_actions = int(num_actions)
        self.n_cos = int(max(1, n_cos))
        self.dueling = bool(dueling)
        self.noisy = bool(noisy)

        hidden = tuple(int(h) for h in hidden_sizes) if hidden_sizes else (128, 128)
        self.embed_dim = int(hidden[-1])
        self.state_hidden = [
            make_dense_layer(h, activation=activation, noisy=self.noisy, name=f"iqn_state_{i}")
            for i, h in enumerate(hidden)
        ]
        self.tau_fc = make_dense_layer(
            self.embed_dim,
            activation=activation,
            noisy=self.noisy,
            name="iqn_tau_fc",
        )
        if self.dueling:
            self.adv_head = make_dense_layer(
                self.num_actions,
                activation=None,
                noisy=self.noisy,
                name="iqn_adv_head",
            )
            self.val_head = make_dense_layer(
                1,
                activation=None,
                noisy=self.noisy,
                name="iqn_val_head",
            )
            self.out_head = None
        else:
            self.out_head = make_dense_layer(
                self.num_actions,
                activation=None,
                noisy=self.noisy,
                name="iqn_q_head",
            )
            self.adv_head = None
            self.val_head = None

        self._pi = tf.constant(np.pi, dtype=tf.float32)
        self._cos_idx = tf.range(1, self.n_cos + 1, dtype=tf.float32)[None, None, :]

    def call(self, x: tf.Tensor, taus: tf.Tensor, training: bool = False) -> tf.Tensor:
        obs = tf.convert_to_tensor(x, tf.float32)
        tau = tf.convert_to_tensor(taus, tf.float32)
        if tau.shape.rank == 1:
            batch_size = tf.shape(obs)[0]
            tau = tf.broadcast_to(tau[None, :], (batch_size, tf.shape(tau)[0]))
        elif tau.shape.rank != 2:
            raise ValueError(f"taus must be rank-1 or rank-2, got shape {tau.shape}")

        z = obs
        for layer in self.state_hidden:
            z = layer(z, training=training)

        tau_expand = tau[:, :, None]
        cos_embed = tf.cos(self._pi * tau_expand * self._cos_idx)
        tau_embed = self.tau_fc(cos_embed, training=training)
        h = tau_embed * z[:, None, :]

        if self.dueling:
            adv = self.adv_head(h, training=training)
            val = self.val_head(h, training=training)
            q = val + (adv - tf.reduce_mean(adv, axis=-1, keepdims=True))
        else:
            q = self.out_head(h, training=training)
        return tf.transpose(q, perm=[0, 2, 1])


class ReplayBuffer:
    """Standard replay buffer for one-step transitions."""

    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)

        self.obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity,), dtype=np.int32)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=np.float32)
        self.discounts = np.zeros((self.capacity,), dtype=np.float32)
        self.size = 0
        self.pos = 0

    def add_batch(
        self,
        obs_batch: np.ndarray,
        act_batch: np.ndarray,
        rew_batch: np.ndarray,
        next_obs_batch: np.ndarray,
        done_batch: np.ndarray,
        disc_batch: np.ndarray,
    ) -> None:
        batch_size = int(obs_batch.shape[0])
        if batch_size <= 0:
            return

        idx = (np.arange(batch_size, dtype=np.int64) + self.pos) % self.capacity
        self.obs[idx] = obs_batch
        self.next_obs[idx] = next_obs_batch
        self.actions[idx] = act_batch
        self.rewards[idx] = rew_batch
        self.dones[idx] = done_batch
        self.discounts[idx] = disc_batch
        self.pos = int((self.pos + batch_size) % self.capacity)
        self.size = int(min(self.capacity, self.size + batch_size))

    def can_sample(self, batch_size: int) -> bool:
        return self.size >= int(batch_size)

    def sample(self, batch_size: int):
        if self.size <= 0:
            raise RuntimeError("ReplayBuffer is empty.")
        b = int(min(self.size, int(batch_size)))
        idx = np.random.randint(0, self.size, size=b)
        return (
            self.obs[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_obs[idx],
            self.dones[idx],
            self.discounts[idx],
        )


class PrioritizedReplayBuffer(ReplayBuffer):
    """Proportional PER buffer."""

    def __init__(self, capacity: int, obs_dim: int, alpha: float = 0.6, eps: float = 1e-6):
        super().__init__(capacity=capacity, obs_dim=obs_dim)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.priorities = np.zeros((self.capacity,), dtype=np.float32)
        self.max_priority = 1.0

    def add_batch(
        self,
        obs_batch: np.ndarray,
        act_batch: np.ndarray,
        rew_batch: np.ndarray,
        next_obs_batch: np.ndarray,
        done_batch: np.ndarray,
        disc_batch: np.ndarray,
    ) -> None:
        batch_size = int(obs_batch.shape[0])
        if batch_size <= 0:
            return
        idx = (np.arange(batch_size, dtype=np.int64) + self.pos) % self.capacity
        super().add_batch(obs_batch, act_batch, rew_batch, next_obs_batch, done_batch, disc_batch)
        self.priorities[idx] = self.max_priority

    def sample(self, batch_size: int, beta: float = 0.4):
        if self.size <= 0:
            raise RuntimeError("PrioritizedReplayBuffer is empty.")
        b = int(min(self.size, int(batch_size)))
        beta = float(beta)

        prios = self.priorities[: self.size] + self.eps
        probs = prios ** self.alpha
        denom = float(probs.sum())
        if denom <= 0.0:
            probs = np.ones((self.size,), dtype=np.float32) / float(self.size)
        else:
            probs = probs / denom
        cdf = np.cumsum(probs)
        idx = np.searchsorted(cdf, np.random.rand(b), side="left")
        idx = np.clip(idx, 0, self.size - 1).astype(np.int32)
        weights = (self.size * probs[idx]) ** (-beta)
        weights = weights / (weights.max() + 1e-8)

        return (
            self.obs[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_obs[idx],
            self.dones[idx],
            self.discounts[idx],
            weights.astype(np.float32),
            idx,
        )

    def update_priorities(self, idx: np.ndarray, priorities: np.ndarray) -> None:
        i = np.asarray(idx, dtype=np.int32)
        p = np.abs(np.asarray(priorities, dtype=np.float32)) + self.eps
        self.priorities[i] = p
        if p.size:
            self.max_priority = max(self.max_priority, float(np.max(p)))


def build_replay_buffer(
    obs_dim: int,
    capacity: int,
    use_per: bool = False,
    per_alpha: float = 0.6,
    per_eps: float = 1e-6,
) -> ReplayBuffer | PrioritizedReplayBuffer:
    """Factory for uniform/PER replay buffers."""
    if bool(use_per):
        return PrioritizedReplayBuffer(
            capacity=int(capacity),
            obs_dim=int(obs_dim),
            alpha=float(per_alpha),
            eps=float(per_eps),
        )
    return ReplayBuffer(capacity=int(capacity), obs_dim=int(obs_dim))


def sample_replay_batch(
    replay: ReplayBuffer | PrioritizedReplayBuffer,
    batch_size: int,
    per_beta: float = 0.4,
) -> ReplaySample:
    """Sample a replay minibatch with unified output for uniform/PER replay."""
    if isinstance(replay, PrioritizedReplayBuffer):
        obs, actions, rewards, next_obs, dones, discounts, is_weights, indices = replay.sample(
            int(batch_size),
            beta=float(per_beta),
        )
        return ReplaySample(
            obs=obs,
            actions=actions,
            rewards=rewards,
            next_obs=next_obs,
            dones=dones,
            discounts=discounts,
            is_weights=is_weights,
            indices=np.asarray(indices, dtype=np.int32),
        )

    obs, actions, rewards, next_obs, dones, discounts = replay.sample(int(batch_size))
    b = int(obs.shape[0])
    return ReplaySample(
        obs=obs,
        actions=actions,
        rewards=rewards,
        next_obs=next_obs,
        dones=dones,
        discounts=discounts,
        is_weights=np.ones((b,), dtype=np.float32),
        indices=None,
    )


class OneStepCancellationEnv:
    """Offline one-step environment built from pre-generated simulation rows."""

    def __init__(
        self,
        states: np.ndarray,
        labels_needed: np.ndarray,
        reward_config: RewardConfig,
        hours: np.ndarray | None = None,
        min_hour: int | None = None,
        max_hour: int | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.states = np.asarray(states, dtype=np.float32)
        self.labels = np.asarray(labels_needed, dtype=np.float32)
        self.reward_config = reward_config
        self.hours = (
            np.asarray(hours, dtype=np.float32)
            if hours is not None
            else np.zeros((self.states.shape[0],), dtype=np.float32)
        )
        self.rng = rng if rng is not None else np.random.default_rng()
        if self.states.ndim != 2:
            raise ValueError(f"states must be 2D, got shape {self.states.shape}")
        if self.labels.ndim != 1:
            raise ValueError(f"labels_needed must be 1D, got shape {self.labels.shape}")
        if self.states.shape[0] != self.labels.shape[0]:
            raise ValueError(
                f"states and labels length mismatch: {self.states.shape[0]} vs {self.labels.shape[0]}"
            )
        if self.hours.shape[0] != self.labels.shape[0]:
            raise ValueError(f"hours and labels length mismatch: {self.hours.shape[0]} vs {self.labels.shape[0]}")
        if min_hour is not None:
            self.min_hour = int(min_hour)
        else:
            self.min_hour = int(np.min(self.hours))
        if max_hour is not None:
            self.max_hour = int(max(1, max_hour))
        else:
            self.max_hour = int(max(1, np.max(self.hours)))

    @property
    def size(self) -> int:
        return int(self.states.shape[0])

    def sample_indices(self, batch_size: int) -> np.ndarray:
        b = int(max(1, batch_size))
        replace = self.size < b
        return self.rng.choice(self.size, size=b, replace=replace).astype(np.int32)

    def collect_rollout(
        self,
        policy: BinaryPolicyNetwork,
        value_fn: ValueNetwork | None,
        batch_size: int,
    ) -> RolloutBatch:
        idx = self.sample_indices(batch_size)
        s = self.states[idx]
        y = self.labels[idx]
        h = self.hours[idx]

        s_tf = tf.convert_to_tensor(s, dtype=tf.float32)
        logits = tf.squeeze(policy(s_tf, training=False), axis=-1).numpy().astype(np.float32)
        if value_fn is None:
            values = np.zeros((s.shape[0],), dtype=np.float32)
        else:
            values = tf.squeeze(value_fn(s_tf, training=False), axis=-1).numpy().astype(np.float32)
        probs = sigmoid_np(logits)
        actions = self.rng.binomial(1, probs).astype(np.float32)

        old_logp = bernoulli_log_prob_np(actions, logits).astype(np.float32)
        rewards = compute_rewards(
            actions,
            y,
            self.reward_config,
            hours=h,
            min_hour=self.min_hour,
            max_hour=self.max_hour,
        ).astype(np.float32)
        returns = rewards.astype(np.float32)
        advantages = (returns - values).astype(np.float32)
        return RolloutBatch(
            states=s,
            labels=y,
            actions=actions,
            old_log_probs=old_logp,
            rewards=rewards,
            hours=h,
            values=values,
            returns=returns,
            advantages=advantages,
        )


class TrajectoryCancellationEnv:
    """Episode-style offline env with irreversible cancellation decisions.

    Each episode corresponds to a route-day sequence ordered by hour.
    Once action=1 (cancel) is taken, the episode terminates.
    """

    def __init__(
        self,
        states: np.ndarray,
        labels_needed: np.ndarray,
        episode_ids: np.ndarray,
        reward_config: RewardConfig,
        hours: np.ndarray | None = None,
        min_hour: int | None = None,
        max_hour: int | None = None,
        gamma: float = 1.0,
        rng: np.random.Generator | None = None,
    ):
        self.states = np.asarray(states, dtype=np.float32)
        self.labels = np.asarray(labels_needed, dtype=np.float32)
        self.episode_ids = np.asarray(episode_ids, dtype=np.int64)
        self.reward_config = reward_config
        self.hours = (
            np.asarray(hours, dtype=np.float32)
            if hours is not None
            else np.zeros((self.states.shape[0],), dtype=np.float32)
        )
        self.gamma = float(max(0.0, gamma))
        self.rng = rng if rng is not None else np.random.default_rng()

        if self.states.ndim != 2:
            raise ValueError(f"states must be 2D, got shape {self.states.shape}")
        if self.labels.ndim != 1:
            raise ValueError(f"labels_needed must be 1D, got shape {self.labels.shape}")
        if self.episode_ids.ndim != 1:
            raise ValueError(f"episode_ids must be 1D, got shape {self.episode_ids.shape}")
        n = self.states.shape[0]
        if self.labels.shape[0] != n or self.hours.shape[0] != n or self.episode_ids.shape[0] != n:
            raise ValueError("states/labels/hours/episode_ids length mismatch")

        if min_hour is not None:
            self.min_hour = int(min_hour)
        else:
            self.min_hour = int(np.min(self.hours))
        if max_hour is not None:
            self.max_hour = int(max(max_hour, 1))
        else:
            self.max_hour = int(max(1, np.max(self.hours)))

        order = np.lexsort((self.hours.astype(np.float32), self.episode_ids.astype(np.int64)))
        ep_sorted = self.episode_ids[order]
        split_points = np.flatnonzero(np.diff(ep_sorted)) + 1
        episode_rows = np.split(order, split_points)
        self.episodes: list[np.ndarray] = [rows.astype(np.int32) for rows in episode_rows if rows.size > 0]
        if not self.episodes:
            raise ValueError("No episodes could be constructed from episode_ids.")

    @property
    def n_episodes(self) -> int:
        return len(self.episodes)

    def sample_episode_indices(self, n_episodes: int) -> np.ndarray:
        n = int(max(1, n_episodes))
        replace = self.n_episodes < n
        return self.rng.choice(self.n_episodes, size=n, replace=replace).astype(np.int32)

    def collect_rollout(
        self,
        policy: BinaryPolicyNetwork,
        value_fn: ValueNetwork | None,
        batch_size: int,
    ) -> RolloutBatch:
        target_steps = int(max(1, batch_size))

        states_out: list[np.ndarray] = []
        labels_out: list[float] = []
        actions_out: list[float] = []
        logp_out: list[float] = []
        rewards_out: list[float] = []
        hours_out: list[float] = []
        values_out: list[float] = []
        returns_out: list[float] = []
        adv_out: list[float] = []

        collected = 0
        while collected < target_steps:
            ep_idx = int(self.rng.integers(0, self.n_episodes))
            row_idx = self.episodes[ep_idx]
            if row_idx.size == 0:
                continue

            ep_states = self.states[row_idx]
            ep_labels = self.labels[row_idx]
            ep_hours = self.hours[row_idx]
            s_tf = tf.convert_to_tensor(ep_states, dtype=tf.float32)
            logits = tf.squeeze(policy(s_tf, training=False), axis=-1).numpy().astype(np.float32)
            if value_fn is None:
                values = np.zeros((row_idx.size,), dtype=np.float32)
            else:
                values = tf.squeeze(value_fn(s_tf, training=False), axis=-1).numpy().astype(np.float32)
            probs = sigmoid_np(logits)

            ep_s: list[np.ndarray] = []
            ep_y: list[float] = []
            ep_a: list[float] = []
            ep_logp: list[float] = []
            ep_r: list[float] = []
            ep_h: list[float] = []
            ep_v: list[float] = []
            ep_done: list[bool] = []

            for t in range(row_idx.size):
                p = float(np.clip(probs[t], 0.0, 1.0))
                action = float(self.rng.binomial(1, p))
                label = float(ep_labels[t])
                hour = float(ep_hours[t])

                is_last = t == (row_idx.size - 1)
                terminal = bool(action >= 0.5) or is_last

                if terminal:
                    reward = float(
                        compute_rewards(
                            np.asarray([action], dtype=np.float32),
                            np.asarray([label], dtype=np.float32),
                            self.reward_config,
                            hours=np.asarray([hour], dtype=np.float32),
                            min_hour=self.min_hour,
                            max_hour=self.max_hour,
                        )[0]
                    )
                else:
                    reward = 0.0

                logp = float(bernoulli_log_prob_np(np.asarray([action], dtype=np.float32), np.asarray([logits[t]], dtype=np.float32))[0])

                ep_s.append(ep_states[t].astype(np.float32))
                ep_y.append(label)
                ep_a.append(action)
                ep_logp.append(logp)
                ep_r.append(reward)
                ep_h.append(hour)
                ep_v.append(float(values[t]))
                ep_done.append(terminal)

                if terminal:
                    break

            if not ep_s:
                continue

            ep_returns = np.zeros((len(ep_r),), dtype=np.float32)
            ret = 0.0
            for t in range(len(ep_r) - 1, -1, -1):
                if ep_done[t]:
                    ret = float(ep_r[t])
                else:
                    ret = float(ep_r[t] + self.gamma * ret)
                ep_returns[t] = ret
            ep_adv = ep_returns - np.asarray(ep_v, dtype=np.float32)

            states_out.extend(ep_s)
            labels_out.extend(ep_y)
            actions_out.extend(ep_a)
            logp_out.extend(ep_logp)
            rewards_out.extend(ep_r)
            hours_out.extend(ep_h)
            values_out.extend(ep_v)
            returns_out.extend(ep_returns.tolist())
            adv_out.extend(ep_adv.tolist())
            collected += len(ep_s)

        return RolloutBatch(
            states=np.asarray(states_out, dtype=np.float32),
            labels=np.asarray(labels_out, dtype=np.float32),
            actions=np.asarray(actions_out, dtype=np.float32),
            old_log_probs=np.asarray(logp_out, dtype=np.float32),
            rewards=np.asarray(rewards_out, dtype=np.float32),
            hours=np.asarray(hours_out, dtype=np.float32),
            values=np.asarray(values_out, dtype=np.float32),
            returns=np.asarray(returns_out, dtype=np.float32),
            advantages=np.asarray(adv_out, dtype=np.float32),
        )


def predict_cancel_probability(
    policy: BinaryPolicyNetwork,
    states: np.ndarray,
    batch_size: int = 8192,
) -> np.ndarray:
    """Predict cancellation probability for all rows."""
    x = np.asarray(states, dtype=np.float32)
    out = np.zeros((x.shape[0],), dtype=np.float32)
    b = int(max(1, batch_size))
    for start in range(0, x.shape[0], b):
        stop = min(x.shape[0], start + b)
        logits = tf.squeeze(policy(x[start:stop], training=False), axis=-1).numpy()
        out[start:stop] = sigmoid_np(logits)
    return out


def to_jsonable(value: Any) -> Any:
    """Convert nested data structures containing numpy/tensors to JSON-ready objects."""
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.integer, np.int32, np.int64)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def save_json(path: str | Path, payload: Any) -> None:
    """Save JSON with stable formatting."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, indent=2, sort_keys=True)


def save_bundle_metadata(path: str | Path, bundle: DataBundle) -> None:
    """Persist dataset/feature metadata for reproducible inference."""
    payload = {
        "state_dim": bundle.state_dim,
        "feature_names": bundle.feature_names,
        "feature_config": asdict(bundle.feature_config),
        "label_config": asdict(bundle.label_config),
        "center_vocab": bundle.center_vocab,
        "dest_vocab": bundle.dest_vocab,
        "normalizer": bundle.normalizer.to_dict(),
        "min_hour": int(bundle.min_hour),
        "max_hour": int(bundle.max_hour),
        "n_train": int(bundle.x_train.shape[0]),
        "n_test": int(bundle.x_test.shape[0]),
    }
    save_json(path, payload)
