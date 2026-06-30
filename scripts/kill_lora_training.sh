#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_NAME="${1:-${RUN_NAME:-qwen3_4b_classification_lora}}"
PID_DIR="${PID_DIR:-outputs/pids}"
PID_FILE="${PID_FILE:-${PID_DIR}/${RUN_NAME}.pid}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-30}"

if [[ ! -f "$PID_FILE" ]]; then
  echo "PID file not found: $PID_FILE"
  echo "Known PID files:"
  find "$PID_DIR" -maxdepth 1 -type f -name '*.pid' -print 2>/dev/null || true
  exit 1
fi

pid="$(cat "$PID_FILE")"
if [[ -z "$pid" ]]; then
  echo "PID file is empty: $PID_FILE"
  rm -f "$PID_FILE"
  exit 1
fi

if ! kill -0 "$pid" 2>/dev/null; then
  echo "Process is not running: pid=$pid"
  rm -f "$PID_FILE"
  exit 0
fi

cmdline="$(tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null || true)"
if [[ "$cmdline" != *"training/train_lora_classification.py"* ]]; then
  echo "Refusing to kill pid=$pid because it does not look like LoRA training."
  echo "cmdline: $cmdline"
  exit 1
fi

echo "Stopping LoRA training: pid=$pid run_name=$RUN_NAME"
kill "$pid"

for ((i = 0; i < TIMEOUT_SECONDS; i++)); do
  if ! kill -0 "$pid" 2>/dev/null; then
    rm -f "$PID_FILE"
    echo "Stopped."
    exit 0
  fi
  sleep 1
done

echo "Process did not exit after ${TIMEOUT_SECONDS}s; sending SIGKILL."
kill -9 "$pid" 2>/dev/null || true
rm -f "$PID_FILE"
echo "Killed."
