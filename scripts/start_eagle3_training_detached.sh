#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
RUN_NAME="${RUN_NAME:-qwen3_4b_email_eagle3}"
LOG_FILE="${LOG_FILE:-outputs/logs/${RUN_NAME}.log}"
PID_FILE="${PID_FILE:-outputs/pids/${RUN_NAME}.pid}"
mkdir -p "$(dirname "$LOG_FILE")" "$(dirname "$PID_FILE")"
if [[ -f "$PID_FILE" ]]; then
  existing_pid="$(cat "$PID_FILE")"
  if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
    echo "EAGLE3 training already running: pid=$existing_pid"; exit 1
  fi
  rm -f "$PID_FILE"
fi
nohup setsid scripts/train_eagle3_online.sh >> "$LOG_FILE" 2>&1 &
pid="$!"
echo "$pid" > "$PID_FILE"
sleep 2
if ! kill -0 "$pid" 2>/dev/null; then
  rm -f "$PID_FILE"
  echo "Process exited during startup. Check log: $LOG_FILE" >&2
  exit 1
fi
echo "Started EAGLE3 training: pid=$pid"
echo "Log: $LOG_FILE"
echo "Stop: scripts/kill_eagle3_training.sh $RUN_NAME"
