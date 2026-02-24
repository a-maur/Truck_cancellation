#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default run settings. Edit here if you want a different baseline run.
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_PATH="${CONFIG_PATH:-${SCRIPT_DIR}/config_ppo.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/disk/lhcb_data/maander/output_truck_cancellation}"
STAGE="${STAGE:-sweep_small}"
TRIAL_INDEX="${TRIAL_INDEX:-0}"
RUN_NAME="${RUN_NAME:-piperun_3}"
WITH_SUMMARY="${WITH_SUMMARY:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
STREAM_TRIAL_LOGS="${STREAM_TRIAL_LOGS:-1}"

if [[ "${WITH_SUMMARY}" == "1" ]]; then
  cmd=(
    "${PYTHON_BIN}"
    "${SCRIPT_DIR}/run_ppo_sweep.py"
    --config "${CONFIG_PATH}"
    --output-root "${OUTPUT_ROOT}"
    --stage "${STAGE}"
    --run-name "${RUN_NAME}"
    --max-trials 1
  )
  if [[ -n "${TRIAL_INDEX}" ]]; then
    cmd+=(--start-index "${TRIAL_INDEX}")
  fi
  if [[ "${SKIP_EXISTING}" == "1" ]]; then
    cmd+=(--skip-existing)
  fi
  if [[ "${STREAM_TRIAL_LOGS}" == "1" ]]; then
    cmd+=(--stream-trial-logs)
  fi
else
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
fi

if [[ "$#" -gt 0 ]]; then
  if [[ "${WITH_SUMMARY}" == "1" ]]; then
    for arg in "$@"; do
      cmd+=(--extra-arg "${arg}")
    done
  else
    cmd+=("$@")
  fi
fi

echo "[run_ppo] ${cmd[*]}"
exec "${cmd[@]}"
