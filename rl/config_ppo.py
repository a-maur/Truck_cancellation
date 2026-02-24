#!/usr/bin/env python3
"""Config helpers for PPO runs and simple sweep expansion.

This mirrors the supply workflow idea:
- keep a stable config file with common args
- optionally define per-stage grids for trial sweeps
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any


def default_config_path() -> Path:
    return Path(__file__).resolve().with_name("config_ppo.json")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path).expanduser().resolve() if path is not None else default_config_path().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"PPO config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid PPO config format at {cfg_path}: expected JSON object.")
    return payload


def _normalize_hidden_sizes_value(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return ",".join(str(int(v)) for v in value)
    return value


def _normalize_args(raw: dict[str, Any]) -> dict[str, Any]:
    out = dict(raw)
    if "hidden_sizes" in out:
        out["hidden_sizes"] = _normalize_hidden_sizes_value(out["hidden_sizes"])
    return out


def expand_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not grid:
        return []
    keys = list(grid.keys())
    values = [list(grid[k]) for k in keys]
    trials: list[dict[str, Any]] = []
    for combo in itertools.product(*values):
        trial = {}
        for key, val in zip(keys, combo):
            trial[key] = val
        trials.append(_normalize_args(trial))
    return trials


def resolve_run_defaults(
    config: dict[str, Any],
    stage: str | None = None,
    trial_index: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve final argparse defaults from config + optional sweep trial."""
    defaults = _normalize_args(config.get("common_args", {}) or {})
    meta: dict[str, Any] = {"stage": None, "trial_index": None, "n_trials": 0}

    stages = config.get("stages", {}) or {}
    if stages:
        if stage is None:
            stage = next(iter(stages.keys()))
        if stage not in stages:
            raise ValueError(f"Unknown stage={stage!r}. Available stages: {list(stages.keys())}")

        stage_cfg = stages[stage] or {}
        defaults.update(_normalize_args(stage_cfg.get("args", {}) or {}))
        trials = expand_grid(stage_cfg.get("grid", {}) or {})
        meta["stage"] = stage
        meta["n_trials"] = len(trials)

        if trial_index is not None:
            idx = int(trial_index)
            if idx < 0 or idx >= len(trials):
                raise IndexError(f"trial_index={idx} out of range [0, {max(0, len(trials)-1)}]")
            defaults.update(trials[idx])
            meta["trial_index"] = idx
            meta["trial_params"] = trials[idx]

    return defaults, meta


def list_stage_trials(
    config: dict[str, Any],
    stage: str | None = None,
) -> tuple[str | None, list[dict[str, Any]], list[str]]:
    stages = config.get("stages", {}) or {}
    if not stages:
        return None, [], []
    if stage is None:
        stage = next(iter(stages.keys()))
    if stage not in stages:
        raise ValueError(f"Unknown stage={stage!r}. Available stages: {list(stages.keys())}")

    stage_cfg = stages[stage] or {}
    trials = expand_grid(stage_cfg.get("grid", {}) or {})
    tag_keys = list(stage_cfg.get("tag_keys", []) or [])
    return stage, trials, tag_keys
