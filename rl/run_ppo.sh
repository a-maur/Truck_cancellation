#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Uses active env python; falls back to python3 when needed.
if [[ -n "${PYTHON_BIN:-}" ]]; then
  _python_bin="${PYTHON_BIN}"
elif command -v python >/dev/null 2>&1; then
  _python_bin="python"
elif command -v python3 >/dev/null 2>&1; then
  _python_bin="python3"
else
  echo "[run_ppo] ERROR: no python executable found (python/python3)." >&2
  exit 127
fi

# Fast dependency preflight so crashes are clear before starting the run.
if ! "${_python_bin}" -c "import numpy" >/dev/null 2>&1; then
  echo "[run_ppo] ERROR: python environment failed to import numpy." >&2
  echo "[run_ppo] This is an environment issue (often missing libgfortran/openblas in conda)." >&2
  echo "[run_ppo] Check with: ${_python_bin} -c 'import numpy; print(numpy.__version__)'" >&2
  echo "[run_ppo] If using OTenv, try reinstalling core deps:" >&2
  echo "  conda install -n OTenv -c conda-forge libgfortran5 libopenblas numpy" >&2
  echo "[run_ppo] Or run from a working env, e.g. rllib_env." >&2
  exit 1
fi

# Default run settings: generate/reuse synthetic data, run PPO sweep, and summarize.
CONFIG_PATH="${CONFIG_PATH:-${SCRIPT_DIR}/config_ppo.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/outputs}"
RUN_NAME="${RUN_NAME:-initial_tests_ppo_newrew_newplots}"
STAGE="${STAGE:-sweep_wide}"
TRIAL_INDEX="${TRIAL_INDEX:-0}"
MAX_TRIALS="${MAX_TRIALS:-2}"
SHUFFLE_TRIALS="${SHUFFLE_TRIALS:-0}"
SHUFFLE_SEED="${SHUFFLE_SEED:-42}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
STREAM_TRIAL_LOGS="${STREAM_TRIAL_LOGS:-1}"
OVERWRITE_DATA="${OVERWRITE_DATA:-0}"

cmd=(
  "${_python_bin}"
  "${SCRIPT_DIR}/run_ppo_sweep.py"
  --config "${CONFIG_PATH}"
  --output-root "${OUTPUT_ROOT}"
  --stage "${STAGE}"
  --run-name "${RUN_NAME}"
  --start-index "${TRIAL_INDEX}"
  --max-trials "${MAX_TRIALS}"
)

if [[ "${SKIP_EXISTING}" == "1" ]]; then
  cmd+=(--skip-existing)
fi

if [[ "${STREAM_TRIAL_LOGS}" == "1" ]]; then
  cmd+=(--stream-trial-logs)
fi

if [[ "${OVERWRITE_DATA}" == "1" ]]; then
  cmd+=(--overwrite-data)
fi

if [[ "${SHUFFLE_TRIALS}" == "1" ]]; then
  cmd+=(--shuffle-trials --shuffle-seed "${SHUFFLE_SEED}")
fi

# Optional simulation overrides via environment variables.
if [[ -n "${SIM_CORRELATION_DEST:-}" ]]; then
  cmd+=(--sim-correlation-dest "${SIM_CORRELATION_DEST}")
fi
if [[ -n "${SIM_CORRELATION_TYPE:-}" ]]; then
  cmd+=(--sim-correlation-type "${SIM_CORRELATION_TYPE}")
fi
if [[ -n "${SIM_N_WEEKS:-}" ]]; then
  cmd+=(--sim-n-weeks "${SIM_N_WEEKS}")
fi
if [[ -n "${SIM_N_WEEKS_HIGH_SEASON:-}" ]]; then
  cmd+=(--sim-n-weeks-high-season "${SIM_N_WEEKS_HIGH_SEASON}")
fi
if [[ -n "${SIM_MARGIN:-}" ]]; then
  cmd+=(--sim-margin "${SIM_MARGIN}")
fi
if [[ -n "${SIM_TRAIN_TEST_RATIO:-}" ]]; then
  cmd+=(--sim-train-test-ratio "${SIM_TRAIN_TEST_RATIO}")
fi
if [[ -n "${SIM_RANDOM_SEED:-}" ]]; then
  cmd+=(--sim-random-seed "${SIM_RANDOM_SEED}")
fi
if [[ -n "${SIM_N_PARCELS_PER_TRUCK:-}" ]]; then
  cmd+=(--sim-n-parcels-per-truck "${SIM_N_PARCELS_PER_TRUCK}")
fi

if [[ "$#" -gt 0 ]]; then
  cmd+=("$@")
fi

echo "[run_ppo] ${cmd[*]}"
exec "${cmd[@]}"
