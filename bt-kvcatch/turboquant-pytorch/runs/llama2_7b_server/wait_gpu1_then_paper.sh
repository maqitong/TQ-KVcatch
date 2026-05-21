#!/usr/bin/env bash
# Wait for GPU1 LongBench main job, then run paper LongBench supplement.
set -eo pipefail
DIR="$(dirname "$0")"
GPU1_PID="${1:-}"

if [[ -n "$GPU1_PID" ]]; then
  echo "[wait] GPU1 pid=$GPU1_PID ..."
  while kill -0 "$GPU1_PID" 2>/dev/null; do
    sleep 120
  done
  echo "[wait] GPU1 main LongBench exited"
fi

bash "$DIR/run_gpu1_paper_longbench.sh"
