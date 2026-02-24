# RL folder

This folder contains RL training code for the truck-cancellation problem in `Truck_cancellation/toy_sim`.

## Files

- `base.py`: shared utilities used by optimisers
  - data loading and feature construction
  - label derivation (`fill_threshold` criterion supported)
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
  - dataset label (`last_truck_needed`) or
  - fill threshold criterion (`needed if estimated last-truck fill >= threshold`, default `0.2`)
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
PYTHON_BIN, CONFIG_PATH, OUTPUT_ROOT, STAGE, TRIAL_INDEX, RUN_NAME
```

Edit these at the top of `run_ppo.sh` if needed.

Useful options:

```bash
python3.11 Truck_cancellation/rl/optimiser_ppo.py \
  --config Truck_cancellation/rl/config_ppo.json \
  --stage sweep_small \
  --trial-index 0 \
  --updates 500 \
  --label-source fill_threshold \
  --needed-fill-threshold 0.2 \
  --early-cancel-bonus 0.7 \
  --early-cancel-penalty 0.7
```

Outputs are saved under `/disk/lhcb_data/maander/output_truck_cancellation/ppo_<timestamp>/` by default (or `--output-dir` if provided), including:

- `policy.weights.h5`
- `value.weights.h5`
- `final_metrics.json`
- `history.json`
- `run_config.json`
- `dataset_metadata.json`

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

The sweep runner writes the following structure under `/disk/lhcb_data/maander/output_truck_cancellation/<run-name>/`:

- `trials/`: trial folders (`trial_XXX_*`) with each trial's PPO artifacts
- `logs/`: one log file per trial
- `summary/`:
  - `summary.json`
  - `summary.csv`
  - `hyperparams_table.tex` (fixed + swept ranges + best values)
  - `hyperparams_table.pdf` (auto-generated when `pdflatex` is installed)
  - `hyperparams_table_pdflatex.log` (compile log when PDF generation is attempted)
  - `plots/objective_by_trial.png`
  - `plots/metrics_by_trial.png`

Ranking defaults to maximizing:

- `test_deterministic.reward_mean`

and can be changed with:

- `--objective-key`
- `--objective-mode max|min`

If a stage has no grid, the LaTeX table includes fixed hyperparameters only.

If `pdflatex` is not available in your environment, the sweep still writes `hyperparams_table.tex` and skips PDF generation.

## Notes

- `optimiser_iqn.py` and `optimiser_qr.py` are placeholders for next steps.
- Python 3.11 is recommended for this folder.
