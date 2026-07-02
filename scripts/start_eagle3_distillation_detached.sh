#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
RUN_NAME="${RUN_NAME:-qwen3_4b_email_eagle3_distillation}"
GPU_ID="${GPU_ID:-3}"
LOG_FILE="${LOG_FILE:-outputs/logs/${RUN_NAME}.log}"
PID_FILE="${PID_FILE:-outputs/pids/${RUN_NAME}.pid}"
DISTILL_ARGS="${DISTILL_ARGS:-}"
mkdir -p "$(dirname "$LOG_FILE")" "$(dirname "$PID_FILE")"
if [[ -f "$PID_FILE" ]]; then
  existing_pid="$(cat "$PID_FILE")"
  if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
    echo "EAGLE3 distillation already running: pid=$existing_pid"; exit 1
  fi
  rm -f "$PID_FILE"
fi
command=(python training/generate_eagle3_distillation_data.py)
if [[ -n "$DISTILL_ARGS" ]]; then
  # shellcheck disable=SC2206
  extra_args=($DISTILL_ARGS)
  command+=("${extra_args[@]}")
fi
nohup setsid env CUDA_VISIBLE_DEVICES="$GPU_ID" "${command[@]}" >> "$LOG_FILE" 2>&1 &
pid="$!"
echo "$pid" > "$PID_FILE"
echo "Started EAGLE3 distillation: pid=$pid gpu=$GPU_ID"
echo "Log: $LOG_FILE"
echo "Stop: scripts/kill_eagle3_distillation.sh $RUN_NAME"
