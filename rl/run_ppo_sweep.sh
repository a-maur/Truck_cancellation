#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default sweep settings. Edit here if needed.
PYTHON_BIN="python3.11"
CONFIG_PATH="${SCRIPT_DIR}/config_ppo.json"
OUTPUT_ROOT="/disk/lhcb_data/maander/output_truck_cancellation"
STAGE="sweep_small"
RUN_NAME="piperun_1_sweep"
SKIP_EXISTING="1"

cmd=(
  "${PYTHON_BIN}"
  "${SCRIPT_DIR}/run_ppo_sweep.py"
  --config "${CONFIG_PATH}"
  --output-root "${OUTPUT_ROOT}"
  --stage "${STAGE}"
  --run-name "${RUN_NAME}"
)

if [[ "${SKIP_EXISTING}" == "1" ]]; then
  cmd+=(--skip-existing)
fi

if [[ "$#" -gt 0 ]]; then
  cmd+=("$@")
fi

echo "[run_ppo_sweep] ${cmd[*]}"
exec "${cmd[@]}"
