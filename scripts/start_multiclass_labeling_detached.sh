#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
for env_file in .env.local .env; do
  if [[ -f "$env_file" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$env_file"
    set +a
  fi
done
RUN_NAME="${RUN_NAME:-multiclass_consensus_labeling}"
LOG_FILE="${LOG_FILE:-outputs/logs/${RUN_NAME}.log}"
PID_FILE="${PID_FILE:-outputs/pids/${RUN_NAME}.pid}"
LABEL_ARGS="${LABEL_ARGS:-}"
mkdir -p "$(dirname "$LOG_FILE")" "$(dirname "$PID_FILE")"
if [[ -f "$PID_FILE" ]]; then
  existing_pid="$(cat "$PID_FILE")"
  if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
    echo "Multiclass labeling already running: pid=$existing_pid"; exit 1
  fi
  rm -f "$PID_FILE"
fi
command=(python training/label_multiclass_consensus.py)
if [[ -n "$LABEL_ARGS" ]]; then
  # shellcheck disable=SC2206
  extra_args=($LABEL_ARGS)
  command+=("${extra_args[@]}")
fi
nohup setsid "${command[@]}" >> "$LOG_FILE" 2>&1 &
pid="$!"
echo "$pid" > "$PID_FILE"
echo "Started multiclass consensus labeling: pid=$pid"
echo "Log: $LOG_FILE"
echo "Stop: scripts/kill_multiclass_labeling.sh $RUN_NAME"
