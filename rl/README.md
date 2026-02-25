# RL folder

This folder contains RL training code for the truck-cancellation problem in `Truck_cancellation/toy_sim`.

## Files

- `base.py`: shared utilities used by optimisers
  - data loading and feature construction
  - label derivation (defaults to dataset `last_truck_needed`; `fill_threshold` also supported)
  - reward shaping (including early-cancel weighting by timestep)
  - metrics (`cancel_success_count`, `cancel_needed_count`, `cancel_rate`, etc.)
  - shared model and replay components
- `optimiser_ppo.py`: PPO trainer and CLI
- `config_ppo.json`: default PPO config + sweep stage definitions
- `config_ppo.py`: config/sweep helper functions
- `run_ppo_sweep.py`: runs trials from config stages and writes ranked summaries

## Current task setup

- One sample = one decision point (center/dest/hour row in `df_per_dest_*`).
- Action:
  - `0` = keep last truck
  - `1` = cancel last truck
- Label (`truck needed`) can be derived by:
  - dataset label (`last_truck_needed`, default) or
  - fill threshold criterion (`needed if estimated last-truck fill >= threshold`)
- Reward is asymmetric and can include **time weighting**:
  - successful early cancellation gets larger bonus
  - wrong early cancellation gets larger penalty
  - weighting is monotonic with timestep (earlier > later)

## Run PPO

From repo root:

```bash
python3.11 Truck_cancellation/rl/optimiser_ppo.py
```

Or with the wrapper script:

```bash
cd Truck_cancellation/rl
bash run_ppo.sh
```

The wrapper already contains defaults for:

```bash
PYTHON_BIN, CONFIG_PATH, OUTPUT_ROOT, STAGE, TRIAL_INDEX, RUN_NAME, WITH_SUMMARY, SKIP_EXISTING, STREAM_TRIAL_LOGS
```

By default `PYTHON_BIN=python`, so it uses your active conda environment.

Edit these at the top of `run_ppo.sh` if needed.

`run_ppo.sh` defaults to `WITH_SUMMARY=1`, which routes the run through `run_ppo_sweep.py` with `--max-trials 1` (and `--start-index` from `TRIAL_INDEX`). This means you get a `summary/` folder even when running a single trial. Set `WITH_SUMMARY=0` to run `optimiser_ppo.py` directly.

`run_ppo.sh` also defaults to `STREAM_TRIAL_LOGS=1`, so optimiser progress lines (for example `update=10/300`) are printed live even in summary/sweep mode.

You can override wrapper defaults at launch without editing the script. Example direct single trial (no sweep wrapper):

```bash
cd Truck_cancellation/rl
WITH_SUMMARY=0 bash run_ppo.sh
```

Useful options:

```bash
python3.11 Truck_cancellation/rl/optimiser_ppo.py \
  --config Truck_cancellation/rl/config_ppo.json \
  --stage sweep_small \
  --trial-index 0 \
  --updates 500 \
  --label-source dataset_label \
  --early-cancel-bonus 0.7 \
  --early-cancel-penalty 0.7
```

Optional convergence stop (based on actor/critic loss slope):

```bash
python3.11 Truck_cancellation/rl/optimiser_ppo.py \
  --early-stop-enabled \
  --early-stop-warmup 100 \
  --early-stop-window 40 \
  --early-stop-check-every 10 \
  --early-stop-patience 3 \
  --early-stop-actor-slope-threshold 1e-4 \
  --early-stop-critic-slope-threshold 5e-4
```

By default, early stopping is disabled. In that default mode, `training_status.json` should report `executed_updates == requested_updates`.

Outputs are saved under `/disk/lhcb_data/maander/output_truck_cancellation/ppo_<timestamp>/` by default (or `--output-dir` if provided), including:

- `policy.weights.h5`
- `value.weights.h5`
- `final_metrics.json`
- `history.json`
- `training_status.json` (requested vs executed updates, stop reason, last loss slopes)
- `run_config.json`
- `dataset_metadata.json`
- `plots_manifest.json`
- `plots/` (when `matplotlib` is available):
  - `training_curves.png` (reward/accuracy/entropy over updates)
  - `actor_loss_over_updates.png`
  - `critic_loss_over_updates.png`
  - `cancel_behavior_over_updates.png` (cancel rate + cancel success rate over updates, with final test values)
  - `hourly_volume_profile.png` (example route-day cumulative volume + agent decisions by hour + final optimal decision)
  - `hourly_decision_rates_test.png` (test cancel rate and cancel success rate by hour)
  - `cancel_metrics_by_destination_test.png`

When `WITH_SUMMARY=1` in `run_ppo.sh`, outputs follow the sweep layout under `/disk/lhcb_data/maander/output_truck_cancellation/<run-name>/` with `trials/`, `logs/`, and `summary/`.

To save in your shared output location with a named folder:

```bash
python3.11 Truck_cancellation/rl/optimiser_ppo.py \
  --output-root /disk/lhcb_data/maander/output_truck_cancellation \
  --run-name piperun_1
```

This writes to:

- `/disk/lhcb_data/maander/output_truck_cancellation/piperun_1`

## List sweep trials

```bash
python3.11 Truck_cancellation/rl/optimiser_ppo.py --list-trials
```

Or pick a stage:

```bash
python3.11 Truck_cancellation/rl/optimiser_ppo.py --stage sweep_small --list-trials
```

## Run full sweep

```bash
python3.11 Truck_cancellation/rl/run_ppo_sweep.py \
  --config Truck_cancellation/rl/config_ppo.json \
  --stage sweep_small \
  --run-name piperun_1 \
  --skip-existing
```

Or with the wrapper script:

```bash
cd Truck_cancellation/rl
bash run_ppo_sweep.sh
```

`run_ppo_sweep.sh` also has fixed defaults at the top:

- `PYTHON_BIN, CONFIG_PATH, OUTPUT_ROOT, STAGE, RUN_NAME, SKIP_EXISTING`

To stream optimiser logs during sweep execution, add:

- `--stream-trial-logs`

The sweep runner writes the following structure under `/disk/lhcb_data/maander/output_truck_cancellation/<run-name>/`:

- `trials/`: trial folders (`trial_XXX_*`) with each trial's PPO artifacts
  - each trial folder also contains `plots/` + `plots_manifest.json` from `optimiser_ppo.py`
- `logs/`: one log file per trial
- `summary/`:
  - `summary.json`
  - `summary.csv`
  - `hyperparams_table.tex` (fixed + swept ranges + best values)
  - `hyperparams_table.pdf` (auto-generated when `pdflatex` is installed)
  - `hyperparams_table_pdflatex.log` (compile log when PDF generation is attempted)
  - `trial_plots/`: copied per-trial plot folders (`trial_XXX_*/`)
  - `plots/objective_by_trial.png`
  - `plots/metrics_by_trial.png`
  - `plots/best_trial_XXX_*.png` (copied best-trial generation plots)

Ranking defaults to maximizing:

- `test_deterministic.reward_mean`

and can be changed with:

- `--objective-key`
- `--objective-mode max|min`

If a stage has no grid, the LaTeX table includes fixed hyperparameters only.

If `pdflatex` is not available in your environment, the sweep still writes `hyperparams_table.tex` and skips PDF generation.

## Correlation vs Hourly Cancel Plot

To train one PPO model per destination-correlation value and plot hourly fraction of correctly cancelled trucks by route:

```bash
python3.11 Truck_cancellation/rl/run_corr_hourly_cancel_plot.py \
  --all-centers-grid \
  --correlations 0.9 \
  --run-name initial_tests
```

Outputs are written under `rl/outputs/<run-name>/` (for example `rl/outputs/initial_tests/`):

- `trial_1/`, `trial_2/`, ...:
  - per-trial PPO artifacts (`policy.weights.h5`, `value.weights.h5`, `final_metrics.json`, plots, etc.)
  - generated data under `trial_N/data/`
  - `trial_N/hourly_metrics.csv`
- `summary/`:
  - `summary/all_hourly_metrics.csv`
  - `summary/summary_by_trial.csv`
  - `summary/all_centers_hourly_correctly_cancelled_by_corr.pdf` (with `--all-centers-grid`)
- `best_weights/`:
  - copied best-trial `policy.weights.h5` and `value.weights.h5`
  - `best_trial.json` with selection metadata
- `info/run_info.json`: run-level configuration and context

In this plot script, `fraction_correctly_cancelled` is:

- `correct_cancels / cancellations` (per center, destination, hour)

## Notes

- `optimiser_iqn.py` and `optimiser_qr.py` are placeholders for next steps.
- Python 3.11 is recommended for this folder.
