#!/usr/bin/env python3
"""Run PPO sweeps from config_ppo.json and summarize trial results.

Output layout under the selected run directory:
- `data/`: generated synthetic dataset used by trials
- `trials/`: one folder per trial run (model artifacts, metrics, history)
- `logs/`: one log file per trial
- `summary/`: aggregate JSON/CSV, overview plots, LaTeX hyperparameter table
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import random
import shutil
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from .config_ppo import default_config_path, list_stage_trials, load_config
except ImportError:
    from config_ppo import default_config_path, list_stage_trials, load_config


def _stringify(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_stringify(v) for v in value) + "]"
    return str(value)


def _latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "{": r"\{",
        "}": r"\}",
        "$": r"\$",
    }
    out = text
    for key, value in replacements.items():
        out = out.replace(key, value)
    return out


def _sanitize_token(value: Any) -> str:
    text = str(value).strip()
    out = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_"}:
            out.append(ch)
        elif ch == ".":
            out.append("p")
        elif ch == ",":
            out.append("-")
    token = "".join(out)
    return token or "na"


def _make_trial_tag(trial: dict[str, Any], tag_keys: list[str]) -> str:
    if not trial:
        return "default"
    keys = tag_keys or sorted(trial.keys())
    parts: list[str] = []
    for key in keys:
        if key not in trial:
            continue
        parts.append(f"{_sanitize_token(key)}{_sanitize_token(trial[key])}")
    return "_".join(parts) if parts else "default"


def _nested_get(payload: dict[str, Any], key: str) -> Any:
    cur: Any = payload
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _format_minutes_seconds(seconds: float) -> str:
    total_seconds = max(0, int(round(float(seconds))))
    minutes, rem = divmod(total_seconds, 60)
    return f"{minutes}m {rem:02d}s"


def _load_module_from_path(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def _load_toy_modules(toy_sim_dir: Path):
    cfg_module = _load_module_from_path("toy_sim_config_sweep", toy_sim_dir / "config.py")

    # toy_sim/core.py imports `from config import ...`, so temporarily provide that alias.
    prev_config_module = sys.modules.get("config")
    sys.modules["config"] = cfg_module
    try:
        core_module = _load_module_from_path("toy_sim_core_sweep", toy_sim_dir / "core.py")
    finally:
        if prev_config_module is not None:
            sys.modules["config"] = prev_config_module
        else:
            sys.modules.pop("config", None)

    return cfg_module, core_module


def _build_raw_save_columns(
    sorting_centers: list[str],
    parcel_types: list[str],
    n_hours: int,
) -> list[str]:
    save_columns = ["center", "day", "season", "hist_avg_vol_tot", "hist_std_vol_tot"]
    save_columns += [f"vol_h{hour}" for hour in range(int(n_hours))]
    destination_suffixes = [
        "_n_exp_trucks",
        "_frac_last_truck_needed",
        "_hist_avg_vol",
        "_hist_std_vol",
        "_overflow",
        "_last_truck_needed",
    ]
    for dest in sorting_centers:
        for parcel_type in parcel_types:
            save_columns.append(f"{dest}_{parcel_type}")
        for suffix in destination_suffixes:
            save_columns.append(f"{dest}{suffix}")
    return save_columns


def _resolve_data_dir(output_root: Path, data_dir_arg: str | None) -> Path:
    if data_dir_arg is None:
        return (output_root / "data").resolve()

    p = Path(data_dir_arg).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (output_root / p).resolve()


def _float_to_int_if_close(value: float, tol: float = 1e-6) -> int | float:
    rounded = int(round(float(value)))
    if abs(float(value) - float(rounded)) <= float(tol):
        return rounded
    return float(value)


def _summarize_days_from_raw_df(df_raw: pd.DataFrame) -> dict[str, Any]:
    if not isinstance(df_raw, pd.DataFrame):
        raise TypeError("Expected pandas DataFrame for raw dataset summary.")

    center_day_rows = int(df_raw.shape[0])
    centers: list[str] = []
    per_center_counts: dict[str, int] = {}
    if "center" in df_raw.columns:
        center_values = df_raw["center"].astype(str)
        centers = sorted(center_values.unique().tolist())
        counts = center_values.value_counts().sort_index()
        per_center_counts = {str(k): int(v) for k, v in counts.items()}

    n_centers = len(centers)
    inferred_global_days: int | float | None = None
    min_rows_per_center: int | None = None
    max_rows_per_center: int | None = None
    if per_center_counts:
        values = list(per_center_counts.values())
        min_rows_per_center = int(min(values))
        max_rows_per_center = int(max(values))
        inferred_global_days = _float_to_int_if_close(float(np.mean(values)))

    return {
        "source": "raw",
        "center_day_rows": int(center_day_rows),
        "n_centers": int(n_centers),
        "centers": centers,
        "per_center_rows": per_center_counts,
        "min_rows_per_center": min_rows_per_center,
        "max_rows_per_center": max_rows_per_center,
        "inferred_global_days": inferred_global_days,
    }


def _summarize_days_from_macro_dfs(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, Any]:
    if not isinstance(train_df, pd.DataFrame) or not isinstance(test_df, pd.DataFrame):
        raise TypeError("Expected pandas DataFrames for macro dataset summary.")

    macro_df = pd.concat([train_df, test_df], ignore_index=True)
    required_cols = {"center", "dest", "hour"}
    if not required_cols.issubset(set(macro_df.columns)):
        missing = sorted(required_cols - set(macro_df.columns))
        raise ValueError(f"Missing required columns for macro day summary: {missing}")

    macro_df = macro_df.copy()
    macro_df["center"] = macro_df["center"].astype(str)
    macro_df["dest"] = macro_df["dest"].astype(str)
    macro_df["hour"] = pd.to_numeric(macro_df["hour"], errors="coerce")
    macro_df = macro_df.dropna(subset=["hour"])
    if macro_df.empty:
        raise ValueError("Macro dataset has no valid hour values for day summary.")

    centers = sorted(macro_df["center"].unique().tolist())
    per_center_days: dict[str, int | float] = {}
    center_day_rows_total = 0.0
    for center in centers:
        sub = macro_df[macro_df["center"] == center]
        rows_center = float(sub.shape[0])
        n_dest = int(sub["dest"].nunique())
        n_hours = int(sub["hour"].nunique())
        denom = float(n_dest * n_hours)
        if denom <= 0.0:
            continue
        center_days = rows_center / denom
        center_days = _float_to_int_if_close(center_days)
        per_center_days[str(center)] = center_days
        center_day_rows_total += float(center_days)

    n_centers = len(per_center_days)
    inferred_global_days: int | float | None = None
    min_rows_per_center: int | float | None = None
    max_rows_per_center: int | float | None = None
    if per_center_days:
        values = [float(v) for v in per_center_days.values()]
        inferred_global_days = _float_to_int_if_close(float(np.mean(values)))
        min_rows_per_center = _float_to_int_if_close(float(np.min(values)))
        max_rows_per_center = _float_to_int_if_close(float(np.max(values)))

    return {
        "source": "macro_estimate",
        "center_day_rows": _float_to_int_if_close(center_day_rows_total),
        "n_centers": int(n_centers),
        "centers": sorted(per_center_days.keys()),
        "per_center_rows": per_center_days,
        "min_rows_per_center": min_rows_per_center,
        "max_rows_per_center": max_rows_per_center,
        "inferred_global_days": inferred_global_days,
    }


def _summarize_days_from_existing_files(raw_path: Path, train_path: Path, test_path: Path) -> dict[str, Any]:
    if raw_path.exists():
        try:
            raw_df = pd.read_pickle(raw_path)
            if isinstance(raw_df, pd.DataFrame):
                return _summarize_days_from_raw_df(raw_df)
        except Exception as exc:
            print(f"[DATA] Warning: failed to read raw dataset for day summary: {exc}")

    try:
        train_df = pd.read_pickle(train_path)
        test_df = pd.read_pickle(test_path)
        if isinstance(train_df, pd.DataFrame) and isinstance(test_df, pd.DataFrame):
            return _summarize_days_from_macro_dfs(train_df=train_df, test_df=test_df)
    except Exception as exc:
        print(f"[DATA] Warning: failed to estimate days from macro train/test data: {exc}")

    return {
        "source": "unknown",
        "center_day_rows": None,
        "n_centers": None,
        "centers": [],
        "per_center_rows": {},
        "min_rows_per_center": None,
        "max_rows_per_center": None,
        "inferred_global_days": None,
    }


def _log_day_summary(day_summary: dict[str, Any], context: str) -> None:
    center_day_rows = day_summary.get("center_day_rows")
    n_centers = day_summary.get("n_centers")
    inferred_global_days = day_summary.get("inferred_global_days")
    source = day_summary.get("source", "unknown")

    if center_day_rows is None:
        print(f"[DATA] {context}: unable to infer day counts (source={source}).")
        return

    print(
        f"[DATA] {context}: center_day_rows={center_day_rows}, "
        f"n_centers={n_centers}, inferred_global_days={inferred_global_days} "
        f"(source={source})"
    )

    min_rows = day_summary.get("min_rows_per_center")
    max_rows = day_summary.get("max_rows_per_center")
    if min_rows is not None and max_rows is not None:
        print(f"[DATA] {context}: per_center_rows_min={min_rows}, per_center_rows_max={max_rows}")


def _ensure_generated_dataset(
    output_root: Path,
    data_dir_arg: str | None,
    overwrite_data: bool,
    sim_correlation_dest: float,
    sim_correlation_type: float,
    sim_n_weeks: int,
    sim_n_weeks_high_season: int,
    sim_margin: float,
    sim_train_test_ratio: float,
    sim_random_seed: int,
    sim_n_parcels_per_truck: int,
) -> dict[str, Any]:
    data_dir = _resolve_data_dir(output_root=output_root, data_dir_arg=data_dir_arg)
    raw_path = data_dir / "df_raw.pkl"
    train_path = data_dir / "df_per_dest_train.pkl"
    test_path = data_dir / "df_per_dest_test.pkl"
    info_path = data_dir / "dataset_info.json"

    t0 = time.time()
    print(f"[DATA] dataset_dir={data_dir}")
    data_dir.mkdir(parents=True, exist_ok=True)

    required_paths = [train_path, test_path]
    has_existing = all(p.exists() for p in required_paths)
    if has_existing and not overwrite_data:
        elapsed = time.time() - t0
        day_summary = _summarize_days_from_existing_files(
            raw_path=raw_path,
            train_path=train_path,
            test_path=test_path,
        )
        print("[DATA] Found existing train/test dataset files; skipping generation.")
        print(f"[DATA] train_path={train_path}")
        print(f"[DATA] test_path={test_path}")
        _log_day_summary(day_summary=day_summary, context="read_data_days")
        print(f"[DATA] reuse_time={_format_minutes_seconds(elapsed)} ({elapsed:.1f}s)")
        return {
            "data_dir": str(data_dir),
            "raw_path": str(raw_path),
            "train_path": str(train_path),
            "test_path": str(test_path),
            "generated": False,
            "overwritten": False,
            "elapsed_s": float(elapsed),
            "day_summary": day_summary,
        }

    if has_existing and overwrite_data:
        print("[DATA] --overwrite-data set: regenerating existing dataset files.")
    else:
        print("[DATA] No existing dataset found: generating synthetic data.")

    if not (0.0 < float(sim_train_test_ratio) < 1.0):
        raise ValueError("--sim-train-test-ratio must be in (0, 1)")

    repo_root = Path(__file__).resolve().parents[1]
    toy_sim_dir = repo_root / "toy_sim"
    toy_cfg, toy_core = _load_toy_modules(toy_sim_dir)
    toy_cfg.validate_shapes()
    day_cfg = toy_cfg.build_day_configuration()

    print(
        "[DATA] generation_params="
        f"(corr_dest={float(sim_correlation_dest):.3f}, corr_type={float(sim_correlation_type):.3f}, "
        f"n_weeks={int(sim_n_weeks)}, n_weeks_high={int(sim_n_weeks_high_season)}, "
        f"margin={float(sim_margin):.3f}, test_ratio={float(sim_train_test_ratio):.3f}, "
        f"seed={int(sim_random_seed)}, parcels_per_truck={int(sim_n_parcels_per_truck)})"
    )

    np.random.seed(int(sim_random_seed))
    df_raw = toy_core.generate_raw_data(
        data_dict=day_cfg,
        corr_dest=float(sim_correlation_dest),
        corr_type=float(sim_correlation_type),
        n_parcels_per_truck=int(sim_n_parcels_per_truck),
        n_weeks=int(sim_n_weeks),
        n_weeks_high_season=int(sim_n_weeks_high_season),
        margin=float(sim_margin),
    )

    n_rows = int(df_raw.shape[0])
    if n_rows < 2:
        raise ValueError(f"Generated dataframe too small for split: n_rows={n_rows}")

    n_test = int(math.ceil(float(sim_train_test_ratio) * n_rows))
    n_test = max(1, min(n_rows - 1, n_test))
    rng = np.random.default_rng(int(sim_random_seed))
    perm = rng.permutation(n_rows)
    idx_test = perm[:n_test]
    idx_train = perm[n_test:]

    train_df_raw = df_raw.iloc[idx_train]
    test_df_raw = df_raw.iloc[idx_test]
    train_df = toy_core.create_macro_stat_dataset_all_origins(train_df_raw)
    test_df = toy_core.create_macro_stat_dataset_all_origins(test_df_raw)
    day_summary = _summarize_days_from_raw_df(df_raw)

    save_columns = _build_raw_save_columns(
        sorting_centers=list(toy_cfg.SORTING_CENTERS),
        parcel_types=list(toy_cfg.PARCEL_TYPES),
        n_hours=int(toy_core.N_HOURS),
    )
    df_raw[save_columns].to_pickle(raw_path)
    train_df.to_pickle(train_path)
    test_df.to_pickle(test_path)

    elapsed = time.time() - t0
    info_payload = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_dir": str(data_dir),
        "raw_path": str(raw_path),
        "train_path": str(train_path),
        "test_path": str(test_path),
        "generated": True,
        "overwritten": bool(overwrite_data and has_existing),
        "elapsed_s": float(elapsed),
        "elapsed_human": _format_minutes_seconds(elapsed),
        "simulation": {
            "correlation_dest": float(sim_correlation_dest),
            "correlation_type": float(sim_correlation_type),
            "n_weeks": int(sim_n_weeks),
            "n_weeks_high_season": int(sim_n_weeks_high_season),
            "margin": float(sim_margin),
            "train_test_ratio": float(sim_train_test_ratio),
            "random_seed": int(sim_random_seed),
            "n_parcels_per_truck": int(sim_n_parcels_per_truck),
        },
        "counts": {
            "raw_rows": int(df_raw.shape[0]),
            "train_rows": int(train_df.shape[0]),
            "test_rows": int(test_df.shape[0]),
        },
        "day_summary": day_summary,
    }
    _write_json(info_path, info_payload)

    print(f"[DATA] Wrote raw_path={raw_path}")
    print(f"[DATA] Wrote train_path={train_path}")
    print(f"[DATA] Wrote test_path={test_path}")
    print(f"[DATA] Wrote info={info_path}")
    _log_day_summary(day_summary=day_summary, context="generated_data_days")
    print(f"[DATA] generation_time={_format_minutes_seconds(elapsed)} ({elapsed:.1f}s)")

    return {
        "data_dir": str(data_dir),
        "raw_path": str(raw_path),
        "train_path": str(train_path),
        "test_path": str(test_path),
        "generated": True,
        "overwritten": bool(overwrite_data and has_existing),
        "elapsed_s": float(elapsed),
        "day_summary": day_summary,
    }


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "rank",
        "trial_index",
        "tag",
        "status",
        "objective",
        "runtime_s",
        "return_code",
        "run_dir",
        "reward_mean_test_det",
        "accuracy_test_det",
        "cancel_rate_test_det",
        "cancel_success_count_test_det",
        "cancel_needed_count_test_det",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def _write_hparam_latex_table(
    path: Path,
    cfg: dict[str, Any],
    stage_name: str,
    trials: list[dict[str, Any]],
    best_trial_index: int | None,
) -> None:
    """Create LaTeX table with fixed/swept hyperparameters and best values."""
    stages = cfg.get("stages", {}) or {}
    stage_cfg = stages.get(stage_name, {}) if stage_name in stages else {}
    common_args = dict(cfg.get("common_args", {}) or {})
    stage_args = dict(stage_cfg.get("args", {}) or {})
    grid = dict(stage_cfg.get("grid", {}) or {})

    base_args = {}
    base_args.update(common_args)
    base_args.update(stage_args)
    swept_keys = set(grid.keys())

    best_trial_params: dict[str, Any] = {}
    if best_trial_index is not None and 0 <= best_trial_index < len(trials):
        best_trial_params = dict(trials[best_trial_index])

    all_keys = sorted(set(base_args.keys()) | swept_keys | set(best_trial_params.keys()))
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\begin{tabular}{|l|l|l|l|}",
        r"\hline",
        r"Parameter & Type & Range/Value & Best \\",
        r"\hline",
    ]
    for key in all_keys:
        if key in swept_keys:
            kind = "swept"
            range_or_value = _stringify(grid.get(key))
            best_value = _stringify(best_trial_params.get(key, base_args.get(key, "")))
        else:
            kind = "fixed"
            fixed_val = base_args.get(key, best_trial_params.get(key, ""))
            range_or_value = _stringify(fixed_val)
            best_value = _stringify(fixed_val)
        lines.append(
            f"{_latex_escape(str(key))} & "
            f"{_latex_escape(kind)} & "
            f"{_latex_escape(range_or_value)} & "
            f"{_latex_escape(best_value)} \\\\"
        )
        lines.append(r"\hline")
    lines += [
        r"\end{tabular}",
        rf"\caption{{PPO hyperparameters for stage {_latex_escape(stage_name)}}}",
        r"\end{table}",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _write_hparam_latex_document(path: Path, table_filename: str = "hyperparams_table.tex") -> None:
    """Write a standalone LaTeX document that inputs the generated table."""
    lines = [
        r"\documentclass{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\begin{document}",
        rf"\input{{{table_filename}}}",
        r"\end{document}",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _compile_hparam_latex_pdf(
    summary_dir: Path,
    table_filename: str = "hyperparams_table.tex",
) -> dict[str, Any]:
    """Compile generated LaTeX table into PDF when pdflatex is available."""
    result: dict[str, Any] = {
        "status": "skipped_missing_pdflatex",
        "pdf_path": str(summary_dir / "hyperparams_table.pdf"),
        "source_table_tex": str(summary_dir / table_filename),
        "document_tex": str(summary_dir / "hyperparams_table_full.tex"),
        "compile_log": str(summary_dir / "hyperparams_table_pdflatex.log"),
    }
    pdflatex_bin = shutil.which("pdflatex")
    if pdflatex_bin is None:
        return result

    doc_path = summary_dir / "hyperparams_table_full.tex"
    _write_hparam_latex_document(doc_path, table_filename=table_filename)
    cmd = [
        str(pdflatex_bin),
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-jobname",
        "hyperparams_table",
        doc_path.name,
    ]

    log_path = summary_dir / "hyperparams_table_pdflatex.log"
    try:
        proc = subprocess.run(cmd, cwd=str(summary_dir), capture_output=True, text=True)
    except Exception as exc:
        with log_path.open("w", encoding="utf-8") as f:
            f.write(f"Failed to run pdflatex: {exc}\n")
        result["status"] = "failed_to_start"
        return result

    with log_path.open("w", encoding="utf-8") as f:
        f.write("Command:\n")
        f.write(" ".join(shlex.quote(x) for x in cmd))
        f.write("\n\n[stdout]\n")
        f.write(proc.stdout or "")
        f.write("\n\n[stderr]\n")
        f.write(proc.stderr or "")

    pdf_path = summary_dir / "hyperparams_table.pdf"
    if proc.returncode == 0 and pdf_path.exists():
        result["status"] = "ok"
    else:
        result["status"] = "failed"
        result["return_code"] = int(proc.returncode)
    return result


def _make_overview_plots(
    summary_dir: Path,
    rows: list[dict[str, Any]],
    objective_mode: str,
) -> None:
    """Write overview plots into summary/plots if matplotlib is available."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    valid = [r for r in rows if r.get("objective") is not None]
    if not valid:
        return

    plot_dir = summary_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: objective by trial index.
    idx = [int(r["trial_index"]) for r in valid]
    obj = [float(r["objective"]) for r in valid]
    plt.figure(figsize=(8, 4))
    plt.scatter(idx, obj, s=20)
    plt.xlabel("trial_index")
    plt.ylabel("objective")
    plt.title(f"Objective by Trial ({objective_mode})")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(plot_dir / "objective_by_trial.png", dpi=140)
    plt.close()

    # Plot 2: reward/accuracy/cancel-needed over trial index when available.
    idx2 = []
    reward = []
    acc = []
    bad_cancel = []
    for r in valid:
        rwd = r.get("reward_mean_test_det")
        ac = r.get("accuracy_test_det")
        bad = r.get("cancel_needed_count_test_det")
        if rwd is None or ac is None or bad is None:
            continue
        idx2.append(int(r["trial_index"]))
        reward.append(float(rwd))
        acc.append(float(ac))
        bad_cancel.append(float(bad))
    if idx2:
        fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
        axes[0].plot(idx2, reward, marker="o", linewidth=1)
        axes[0].set_ylabel("reward_mean")
        axes[0].grid(alpha=0.2)
        axes[1].plot(idx2, acc, marker="o", linewidth=1, color="tab:green")
        axes[1].set_ylabel("accuracy")
        axes[1].grid(alpha=0.2)
        axes[2].plot(idx2, bad_cancel, marker="o", linewidth=1, color="tab:red")
        axes[2].set_ylabel("cancel_needed_count")
        axes[2].set_xlabel("trial_index")
        axes[2].grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(plot_dir / "metrics_by_trial.png", dpi=140)
        plt.close(fig)


def _run_trial_command(
    cmd: list[str],
    log_path: Path,
    stream_trial_logs: bool,
) -> int:
    """Run one trial command, persist logs, and optionally stream live output."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"

    if not stream_trial_logs:
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
        with log_path.open("w", encoding="utf-8") as f:
            f.write(proc.stdout or "")
            if proc.stderr:
                f.write("\n[stderr]\n")
                f.write(proc.stderr)
        return int(proc.returncode)

    with log_path.open("w", encoding="utf-8") as f:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        if proc.stdout is not None:
            for line in proc.stdout:
                f.write(line)
                print(line, end="")
        return int(proc.wait())


def _copy_trial_plots_into_summary(
    summary_dir: Path,
    rows: list[dict[str, Any]],
    best_trial_index: int | None,
) -> dict[str, Any]:
    """Copy per-trial run plots into summary folders instead of regenerating them."""
    trial_plots_root = summary_dir / "trial_plots"
    summary_plot_root = summary_dir / "plots"
    trial_plots_root.mkdir(parents=True, exist_ok=True)
    summary_plot_root.mkdir(parents=True, exist_ok=True)

    copied_trials = 0
    copied_files = 0
    missing_plot_dirs: list[str] = []
    copied_by_trial: list[dict[str, Any]] = []

    for row in rows:
        run_dir_raw = row.get("run_dir")
        if not run_dir_raw:
            continue
        trial_idx = int(row.get("trial_index", -1))
        tag = str(row.get("tag", "default"))
        trial_key = f"trial_{trial_idx:03d}_{tag}"
        src_plot_dir = Path(str(run_dir_raw)).expanduser() / "plots"
        src_manifest = Path(str(run_dir_raw)).expanduser() / "plots_manifest.json"
        dst_plot_dir = trial_plots_root / trial_key

        copied_names: list[str] = []
        if src_plot_dir.exists():
            dst_plot_dir.mkdir(parents=True, exist_ok=True)
            for src in sorted(src_plot_dir.glob("*.png")):
                dst = dst_plot_dir / src.name
                try:
                    shutil.copy2(src, dst)
                    copied_names.append(src.name)
                except Exception:
                    continue
        else:
            missing_plot_dirs.append(str(src_plot_dir))

        if src_manifest.exists():
            dst_plot_dir.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(src_manifest, dst_plot_dir / "plots_manifest.json")
            except Exception:
                pass

        if copied_names:
            copied_trials += 1
            copied_files += len(copied_names)
            copied_by_trial.append(
                {
                    "trial_index": trial_idx,
                    "tag": tag,
                    "n_files": len(copied_names),
                    "files": copied_names,
                    "dest_dir": str(dst_plot_dir),
                }
            )

    best_trial_plot_files: list[str] = []
    if best_trial_index is not None:
        best_row = next(
            (
                r
                for r in rows
                if int(r.get("trial_index", -1)) == int(best_trial_index)
                and r.get("run_dir") is not None
            ),
            None,
        )
        if best_row is not None:
            best_run_dir = Path(str(best_row["run_dir"])).expanduser()
            best_plot_dir = best_run_dir / "plots"
            if best_plot_dir.exists():
                for src in sorted(best_plot_dir.glob("*.png")):
                    dst_name = f"best_trial_{int(best_trial_index):03d}_{src.name}"
                    dst = summary_plot_root / dst_name
                    try:
                        shutil.copy2(src, dst)
                        best_trial_plot_files.append(str(dst))
                    except Exception:
                        continue

    return {
        "trial_plots_root": str(trial_plots_root),
        "copied_trials": int(copied_trials),
        "copied_files": int(copied_files),
        "missing_plot_dirs": sorted(set(missing_plot_dirs)),
        "copied_by_trial": copied_by_trial,
        "best_trial_plot_files": best_trial_plot_files,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run PPO sweep trials and summarize results.")
    parser.add_argument("--config", type=str, default=str(default_config_path()), help="Path to config_ppo.json")
    parser.add_argument("--stage", type=str, default=None, help="Stage name in config (default: first stage)")
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Root directory for sweep outputs (default: config.output_root or rl/outputs)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional named sweep folder inside output root (example: piperun_1)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Synthetic dataset directory. Relative paths resolve under run output root (default: data).",
    )
    parser.add_argument(
        "--overwrite-data",
        action="store_true",
        help="Regenerate synthetic train/test data even if dataset files already exist.",
    )
    parser.add_argument("--sim-correlation-dest", type=float, default=0.9, help="Destination correlation for simulation.")
    parser.add_argument("--sim-correlation-type", type=float, default=0.3, help="Parcel-type correlation for simulation.")
    parser.add_argument("--sim-n-weeks", type=int, default=100, help="Number of low-season weeks to simulate.")
    parser.add_argument(
        "--sim-n-weeks-high-season",
        type=int,
        default=10,
        help="Number of high-season weeks to simulate.",
    )
    parser.add_argument("--sim-margin", type=float, default=0.0, help="Need/overflow margin used in simulation.")
    parser.add_argument(
        "--sim-train-test-ratio",
        type=float,
        default=0.2,
        help="Fraction of generated daily rows assigned to test split.",
    )
    parser.add_argument("--sim-random-seed", type=int, default=42, help="Random seed for simulation and split.")
    parser.add_argument(
        "--sim-n-parcels-per-truck",
        type=int,
        default=100,
        help="Truck capacity used during data generation.",
    )
    parser.add_argument(
        "--python-bin",
        type=str,
        default=None,
        help="Python executable to run optimiser_ppo.py (default: config.python_bin or sys.executable)",
    )
    parser.add_argument("--start-index", type=int, default=0, help="Start trial index (inclusive)")
    parser.add_argument("--max-trials", type=int, default=None, help="Max number of trials to run")
    parser.add_argument("--shuffle-trials", action="store_true", help="Shuffle trial order")
    parser.add_argument("--shuffle-seed", type=int, default=42, help="Seed used when shuffling")
    parser.add_argument("--skip-existing", action="store_true", help="Skip trials with existing final_metrics.json")
    parser.add_argument(
        "--stream-trial-logs",
        action="store_true",
        help="Stream optimiser logs to console while also writing per-trial log files",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands only, do not execute")
    parser.add_argument(
        "--objective-key",
        type=str,
        default="test_deterministic.reward_mean",
        help="Nested key in final_metrics.json used for ranking",
    )
    parser.add_argument(
        "--objective-mode",
        type=str,
        default="max",
        choices=["max", "min"],
        help="Whether objective should be maximized or minimized",
    )
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Additional argument passed to optimiser_ppo.py (repeatable)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args, passthrough_args = parser.parse_known_args()

    config_path = Path(args.config).expanduser().resolve()
    cfg = load_config(config_path)

    stage_name, trials, tag_keys = list_stage_trials(cfg, stage=args.stage)
    if stage_name is None:
        stage_name = "default"
        trials = [{}]
        tag_keys = []
    if not trials:
        trials = [{}]

    start = max(0, int(args.start_index))
    indices = list(range(start, len(trials)))
    if args.shuffle_trials:
        rng = random.Random(int(args.shuffle_seed))
        rng.shuffle(indices)
    if args.max_trials is not None:
        indices = indices[: max(0, int(args.max_trials))]

    script_path = Path(__file__).resolve().with_name("optimiser_ppo.py")
    cfg_output_root = cfg.get("output_root")
    if args.output_root is not None:
        base_output_root = Path(args.output_root).expanduser().resolve()
    elif cfg_output_root:
        cfg_root = Path(str(cfg_output_root)).expanduser()
        if not cfg_root.is_absolute():
            cfg_root = config_path.parent / cfg_root
        base_output_root = cfg_root.resolve()
    else:
        base_output_root = (Path(__file__).resolve().parent / "outputs").resolve()
    output_root = base_output_root / (str(args.run_name) if args.run_name else stage_name)
    trials_root = output_root / "trials"
    logs_root = output_root / "logs"
    summary_dir = output_root / "summary"
    trials_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    python_bin = args.python_bin or cfg.get("python_bin") or sys.executable
    extra_args = list(args.extra_arg or [])
    if passthrough_args:
        extra_args.extend(passthrough_args)

    print(
        f"[SWEEP] config={config_path} stage={stage_name} "
        f"n_trials_total={len(trials)} n_selected={len(indices)}"
    )
    print(f"[SWEEP] output_root={output_root}")
    print(f"[SWEEP] python={python_bin}")
    if passthrough_args:
        print(f"[SWEEP] forwarding_unknown_args_to_optimiser={passthrough_args}")

    rows: list[dict[str, Any]] = []
    t0 = time.time()
    if args.dry_run:
        data_dir = _resolve_data_dir(output_root=output_root, data_dir_arg=args.data_dir)
        data_info = {
            "data_dir": str(data_dir),
            "raw_path": str(data_dir / "df_raw.pkl"),
            "train_path": str(data_dir / "df_per_dest_train.pkl"),
            "test_path": str(data_dir / "df_per_dest_test.pkl"),
            "generated": False,
            "overwritten": False,
            "elapsed_s": 0.0,
            "day_summary": {
                "source": "dry_run",
                "center_day_rows": None,
                "n_centers": None,
                "centers": [],
                "per_center_rows": {},
                "min_rows_per_center": None,
                "max_rows_per_center": None,
                "inferred_global_days": None,
            },
        }
        print("[DATA] dry_run enabled: skipping dataset generation/reuse checks.")
        print(f"[DATA] expected_train_path={data_info['train_path']}")
        print(f"[DATA] expected_test_path={data_info['test_path']}")
    else:
        data_info = _ensure_generated_dataset(
            output_root=output_root,
            data_dir_arg=args.data_dir,
            overwrite_data=bool(args.overwrite_data),
            sim_correlation_dest=float(args.sim_correlation_dest),
            sim_correlation_type=float(args.sim_correlation_type),
            sim_n_weeks=int(args.sim_n_weeks),
            sim_n_weeks_high_season=int(args.sim_n_weeks_high_season),
            sim_margin=float(args.sim_margin),
            sim_train_test_ratio=float(args.sim_train_test_ratio),
            sim_random_seed=int(args.sim_random_seed),
            sim_n_parcels_per_truck=int(args.sim_n_parcels_per_truck),
        )

    for rank_in_order, trial_idx in enumerate(indices, start=1):
        trial_cfg = trials[trial_idx]
        tag = _make_trial_tag(trial_cfg, tag_keys)
        run_dir = trials_root / f"trial_{trial_idx:03d}_{tag}"
        metrics_path = run_dir / "final_metrics.json"
        log_path = logs_root / f"trial_{trial_idx:03d}_{tag}.log"

        cmd = [
            str(python_bin),
            str(script_path),
            "--config",
            str(config_path),
            "--stage",
            str(stage_name),
            "--trial-index",
            str(trial_idx),
            "--train-path",
            str(data_info["train_path"]),
            "--test-path",
            str(data_info["test_path"]),
            "--output-dir",
            str(run_dir),
        ] + extra_args

        record: dict[str, Any] = {
            "rank": None,
            "trial_index": trial_idx,
            "tag": tag,
            "status": "pending",
            "objective": None,
            "runtime_s": 0.0,
            "return_code": None,
            "run_dir": str(run_dir),
            "train_path": str(data_info["train_path"]),
            "test_path": str(data_info["test_path"]),
            "command": cmd,
        }

        if args.skip_existing and metrics_path.exists():
            record["status"] = "skipped_existing"
            metrics = _load_json(metrics_path)
            if metrics is not None:
                record["objective"] = _nested_get(metrics, args.objective_key)
                test_det = metrics.get("test_deterministic", {})
                if isinstance(test_det, dict):
                    record["reward_mean_test_det"] = test_det.get("reward_mean")
                    record["accuracy_test_det"] = test_det.get("decision_accuracy")
                    record["cancel_rate_test_det"] = test_det.get("cancel_rate")
                    record["cancel_success_count_test_det"] = test_det.get("cancel_success_count")
                    record["cancel_needed_count_test_det"] = test_det.get("cancel_needed_count")
            rows.append(record)
            print(f"[SWEEP] [{rank_in_order}/{len(indices)}] skip trial={trial_idx} ({tag})")
            continue

        print(f"[SWEEP] [{rank_in_order}/{len(indices)}] run trial={trial_idx} ({tag})")
        print(f"[SWEEP] cmd: {' '.join(shlex.quote(x) for x in cmd)}")
        if args.dry_run:
            record["status"] = "dry_run"
            rows.append(record)
            continue

        run_dir.mkdir(parents=True, exist_ok=True)
        t_start = time.time()
        return_code = _run_trial_command(
            cmd=cmd,
            log_path=log_path,
            stream_trial_logs=bool(args.stream_trial_logs),
        )
        runtime_s = time.time() - t_start

        record["runtime_s"] = runtime_s
        record["return_code"] = int(return_code)
        record["status"] = "ok" if return_code == 0 else "failed"

        metrics = _load_json(metrics_path)
        if metrics is not None:
            record["objective"] = _nested_get(metrics, args.objective_key)
            test_det = metrics.get("test_deterministic", {})
            if isinstance(test_det, dict):
                record["reward_mean_test_det"] = test_det.get("reward_mean")
                record["accuracy_test_det"] = test_det.get("decision_accuracy")
                record["cancel_rate_test_det"] = test_det.get("cancel_rate")
                record["cancel_success_count_test_det"] = test_det.get("cancel_success_count")
                record["cancel_needed_count_test_det"] = test_det.get("cancel_needed_count")

        rows.append(record)
        print(
            f"[SWEEP] trial={trial_idx} status={record['status']} "
            f"objective={record['objective']} "
            f"runtime={_format_minutes_seconds(runtime_s)} ({runtime_s:.1f}s)"
        )

    valid_rows = [r for r in rows if r.get("objective") is not None]
    reverse = args.objective_mode == "max"
    valid_rows_sorted = sorted(valid_rows, key=lambda r: float(r["objective"]), reverse=reverse)
    rank_map = {id(row): i + 1 for i, row in enumerate(valid_rows_sorted)}
    for row in rows:
        row["rank"] = rank_map.get(id(row))

    elapsed_total = time.time() - t0
    best_trial_index = int(valid_rows_sorted[0]["trial_index"]) if valid_rows_sorted else None
    summary = {
        "config": str(config_path),
        "stage": stage_name,
        "objective_key": args.objective_key,
        "objective_mode": args.objective_mode,
        "n_trials_total": len(trials),
        "n_trials_selected": len(indices),
        "elapsed_s": float(elapsed_total),
        "elapsed_human": _format_minutes_seconds(elapsed_total),
        "output_root": str(output_root),
        "dataset": {
            "data_dir": str(data_info["data_dir"]),
            "raw_path": str(data_info["raw_path"]),
            "train_path": str(data_info["train_path"]),
            "test_path": str(data_info["test_path"]),
            "generated": bool(data_info["generated"]),
            "overwritten": bool(data_info["overwritten"]),
            "elapsed_s": float(data_info["elapsed_s"]),
            "elapsed_human": _format_minutes_seconds(float(data_info["elapsed_s"])),
            "overwrite_data_requested": bool(args.overwrite_data),
            "sim_correlation_dest": float(args.sim_correlation_dest),
            "sim_correlation_type": float(args.sim_correlation_type),
            "sim_n_weeks": int(args.sim_n_weeks),
            "sim_n_weeks_high_season": int(args.sim_n_weeks_high_season),
            "sim_margin": float(args.sim_margin),
            "sim_train_test_ratio": float(args.sim_train_test_ratio),
            "sim_random_seed": int(args.sim_random_seed),
            "sim_n_parcels_per_truck": int(args.sim_n_parcels_per_truck),
            "day_summary": data_info.get("day_summary"),
        },
        "trials_dir": str(trials_root),
        "logs_dir": str(logs_root),
        "summary_dir": str(summary_dir),
        "best_trial_index": best_trial_index,
        "best_objective": (float(valid_rows_sorted[0]["objective"]) if valid_rows_sorted else None),
        "rows": rows,
    }

    _write_summary_csv(summary_dir / "summary.csv", rows)
    _write_hparam_latex_table(
        path=summary_dir / "hyperparams_table.tex",
        cfg=cfg,
        stage_name=stage_name,
        trials=trials,
        best_trial_index=best_trial_index,
    )
    latex_pdf = _compile_hparam_latex_pdf(summary_dir=summary_dir, table_filename="hyperparams_table.tex")
    summary["latex_pdf"] = latex_pdf
    copied_trial_plots = _copy_trial_plots_into_summary(
        summary_dir=summary_dir,
        rows=rows,
        best_trial_index=best_trial_index,
    )
    summary["copied_trial_plots"] = copied_trial_plots
    _write_json(summary_dir / "summary.json", summary)
    _make_overview_plots(summary_dir=summary_dir, rows=rows, objective_mode=args.objective_mode)

    print(f"[SWEEP] Wrote summary: {summary_dir / 'summary.json'}")
    print(f"[SWEEP] Wrote summary: {summary_dir / 'summary.csv'}")
    print(f"[SWEEP] Wrote summary: {summary_dir / 'hyperparams_table.tex'}")
    if latex_pdf.get("status") == "ok":
        print(f"[SWEEP] Wrote summary: {summary_dir / 'hyperparams_table.pdf'}")
    elif latex_pdf.get("status") == "skipped_missing_pdflatex":
        print("[SWEEP] pdflatex not found; skipping PDF generation (LaTeX .tex kept).")
    else:
        print(
            "[SWEEP] Failed to compile LaTeX PDF; "
            f"see {summary_dir / 'hyperparams_table_pdflatex.log'}"
        )
    print(
        "[SWEEP] Copied trial plots: "
        f"n_trials={copied_trial_plots.get('copied_trials', 0)} "
        f"n_files={copied_trial_plots.get('copied_files', 0)} "
        f"into {summary_dir / 'trial_plots'}"
    )
    print(f"[SWEEP] Wrote plots in: {summary_dir / 'plots'}")
    print(f"[SWEEP] Total runtime: {_format_minutes_seconds(elapsed_total)} ({elapsed_total:.1f}s)")
    if valid_rows_sorted:
        best = valid_rows_sorted[0]
        print(
            f"[SWEEP] Best trial={best['trial_index']} rank=1 "
            f"objective={best['objective']} run_dir={best['run_dir']}"
        )


if __name__ == "__main__":
    main()
