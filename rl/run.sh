#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cmd=(bash "${SCRIPT_DIR}/run_ppo.sh")
if [[ "$#" -gt 0 ]]; then
  cmd+=("$@")
fi

echo "[run] ${cmd[*]}"
exec "${cmd[@]}"
