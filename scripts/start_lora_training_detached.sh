#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_NAME="${RUN_NAME:-qwen3_4b_classification_lora}"
GPU_ID="${GPU_ID:-3}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-models/Qwen3-4B}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/lora/${RUN_NAME}}"
LOG_DIR="${LOG_DIR:-outputs/logs}"
PID_DIR="${PID_DIR:-outputs/pids}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_NAME}.log}"
PID_FILE="${PID_FILE:-${PID_DIR}/${RUN_NAME}.pid}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-5000}"
MAX_VALIDATION_SAMPLES="${MAX_VALIDATION_SAMPLES:-1000}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-4096}"
EPOCHS="${EPOCHS:-1.0}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
EVAL_STRATEGY="${EVAL_STRATEGY:-epoch}"
SAVE_STRATEGY="${SAVE_STRATEGY:-epoch}"
EVAL_STEPS="${EVAL_STEPS:-200}"
SAVE_STEPS="${SAVE_STEPS:-200}"
SEED="${SEED:-20260630}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "$LOG_DIR" "$PID_DIR" "$OUTPUT_DIR"

if [[ -f "$PID_FILE" ]]; then
  existing_pid="$(cat "$PID_FILE")"
  if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
    echo "LoRA training already running: pid=$existing_pid pid_file=$PID_FILE"
    echo "Log: $LOG_FILE"
    exit 1
  fi
  rm -f "$PID_FILE"
fi

command=(
  "$PYTHON_BIN" training/train_lora_classification.py
  --model-path "$MODEL_PATH"
  --output-dir "$OUTPUT_DIR"
  --max-train-samples "$MAX_TRAIN_SAMPLES"
  --max-validation-samples "$MAX_VALIDATION_SAMPLES"
  --max-seq-length "$MAX_SEQ_LENGTH"
  --epochs "$EPOCHS"
  --learning-rate "$LEARNING_RATE"
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
  --per-device-train-batch-size "$PER_DEVICE_TRAIN_BATCH_SIZE"
  --per-device-eval-batch-size "$PER_DEVICE_EVAL_BATCH_SIZE"
  --eval-strategy "$EVAL_STRATEGY"
  --save-strategy "$SAVE_STRATEGY"
  --eval-steps "$EVAL_STEPS"
  --save-steps "$SAVE_STEPS"
  --seed "$SEED"
  --load-in-4bit
  --bf16
)

if [[ -n "$EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2206
  extra_args_array=($EXTRA_ARGS)
  command+=("${extra_args_array[@]}")
fi

{
  echo "[$(date -Is)] Starting LoRA training"
  echo "root=$ROOT_DIR"
  echo "run_name=$RUN_NAME"
  echo "gpu_id=$GPU_ID"
  echo "output_dir=$OUTPUT_DIR"
  echo "pid_file=$PID_FILE"
  echo "log_file=$LOG_FILE"
  printf 'command='
  printf '%q ' "${command[@]}"
  echo
  echo
} >> "$LOG_FILE"

CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONPATH=src nohup "${command[@]}" >> "$LOG_FILE" 2>&1 &
pid="$!"
echo "$pid" > "$PID_FILE"

echo "Started LoRA training in background."
echo "PID: $pid"
echo "GPU: $GPU_ID"
echo "Log: $LOG_FILE"
echo "PID file: $PID_FILE"
echo "Follow logs: tail -f $LOG_FILE"
echo "Stop: scripts/kill_lora_training.sh ${RUN_NAME}"
