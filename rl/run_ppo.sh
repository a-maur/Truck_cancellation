#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default run settings. Edit here if you want a different baseline run.
PYTHON_BIN="python3.11"
CONFIG_PATH="${SCRIPT_DIR}/config_ppo.json"
OUTPUT_ROOT="/disk/lhcb_data/maander/output_truck_cancellation"
STAGE="sweep_small"
TRIAL_INDEX="0"
RUN_NAME="piperun_1"

cmd=(
  "${PYTHON_BIN}"
  "${SCRIPT_DIR}/optimiser_ppo.py"
  --config "${CONFIG_PATH}"
  --output-root "${OUTPUT_ROOT}"
  --stage "${STAGE}"
  --run-name "${RUN_NAME}"
)

if [[ -n "${TRIAL_INDEX}" ]]; then
  cmd+=(--trial-index "${TRIAL_INDEX}")
fi

if [[ "$#" -gt 0 ]]; then
  cmd+=("$@")
fi

echo "[run_ppo] ${cmd[*]}"
exec "${cmd[@]}"
