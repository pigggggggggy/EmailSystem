#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
RUN_NAME="${1:-${RUN_NAME:-qwen3_4b_email_eagle3}}"
PID_FILE="${PID_FILE:-outputs/pids/${RUN_NAME}.pid}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-30}"
[[ -f "$PID_FILE" ]] || { echo "PID file not found: $PID_FILE" >&2; exit 1; }
pid="$(cat "$PID_FILE")"
if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then rm -f "$PID_FILE"; echo "Not running."; exit 0; fi
cmdline="$(tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null || true)"
[[ "$cmdline" == *"train_eagle3_online.sh"* ]] || { echo "Refusing to stop unrelated pid=$pid: $cmdline" >&2; exit 1; }
echo "Stopping EAGLE3 process group: $pid"
kill -- "-$pid" 2>/dev/null || kill "$pid"
for ((i=0; i<TIMEOUT_SECONDS; i++)); do
  if ! kill -0 "$pid" 2>/dev/null; then rm -f "$PID_FILE"; echo "Stopped."; exit 0; fi
  sleep 1
done
kill -9 -- "-$pid" 2>/dev/null || kill -9 "$pid" 2>/dev/null || true
rm -f "$PID_FILE"
