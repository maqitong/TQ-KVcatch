#!/usr/bin/env bash
# Wait for main experiment PIDs, then run paper-baseline supplements.
set -eo pipefail
DIR="$(dirname "$0")"
GPU0_PID="${1:-}"
GPU1_PID="${2:-}"

wait_pid() {
  local label="$1" pid="$2"
  if [[ -z "$pid" ]]; then
    echo "[wait] no $label pid, skip wait"
    return
  fi
  echo "[wait] $label pid=$pid ..."
  while kill -0 "$pid" 2>/dev/null; do
    sleep 120
  done
  echo "[wait] $label pid=$pid exited"
}

wait_pid GPU0 "$GPU0_PID"
bash "$DIR/run_gpu0_paper_niah.sh"

wait_pid GPU1 "$GPU1_PID"
bash "$DIR/run_gpu1_paper_longbench.sh"

echo "[wait] all paper supplements done $(date -Iseconds)"
