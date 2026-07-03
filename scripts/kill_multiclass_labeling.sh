#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
RUN_NAME="${1:-${RUN_NAME:-multiclass_consensus_labeling}}"
PID_FILE="${PID_FILE:-outputs/pids/${RUN_NAME}.pid}"
[[ -f "$PID_FILE" ]] || { echo "No PID file: $PID_FILE"; exit 1; }
pid="$(cat "$PID_FILE")"
[[ -n "$pid" ]] || { echo "Empty PID file: $PID_FILE"; exit 1; }
cmdline="$(ps -o args= -p "$pid" 2>/dev/null || true)"
[[ "$cmdline" == *"label_multiclass_consensus.py"* ]] || {
  echo "Refusing to stop unrelated pid=$pid: $cmdline" >&2; exit 1;
}
kill -- -"$pid" 2>/dev/null || kill "$pid"
echo "Stopped multiclass labeling: pid=$pid"
rm -f "$PID_FILE"
