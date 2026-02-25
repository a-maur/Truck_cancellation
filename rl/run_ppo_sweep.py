#!/usr/bin/env python3
"""Run PPO sweeps from config_ppo.json and summarize trial results.

Output layout under the selected run directory:
- `trials/`: one folder per trial run (model artifacts, metrics, history)
- `logs/`: one log file per trial
- `summary/`: aggregate JSON/CSV, overview plots, LaTeX hyperparameter table
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

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
    args = build_parser().parse_args()

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

    print(
        f"[SWEEP] config={config_path} stage={stage_name} "
        f"n_trials_total={len(trials)} n_selected={len(indices)}"
    )
    print(f"[SWEEP] output_root={output_root}")
    print(f"[SWEEP] python={python_bin}")

    rows: list[dict[str, Any]] = []
    t0 = time.time()

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
            f"objective={record['objective']} runtime={runtime_s:.1f}s"
        )

    valid_rows = [r for r in rows if r.get("objective") is not None]
    reverse = args.objective_mode == "max"
    valid_rows_sorted = sorted(valid_rows, key=lambda r: float(r["objective"]), reverse=reverse)
    rank_map = {id(row): i + 1 for i, row in enumerate(valid_rows_sorted)}
    for row in rows:
        row["rank"] = rank_map.get(id(row))

    best_trial_index = int(valid_rows_sorted[0]["trial_index"]) if valid_rows_sorted else None
    summary = {
        "config": str(config_path),
        "stage": stage_name,
        "objective_key": args.objective_key,
        "objective_mode": args.objective_mode,
        "n_trials_total": len(trials),
        "n_trials_selected": len(indices),
        "elapsed_s": time.time() - t0,
        "output_root": str(output_root),
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
    if valid_rows_sorted:
        best = valid_rows_sorted[0]
        print(
            f"[SWEEP] Best trial={best['trial_index']} rank=1 "
            f"objective={best['objective']} run_dir={best['run_dir']}"
        )


if __name__ == "__main__":
    main()
