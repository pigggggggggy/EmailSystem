#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export RUN_NAME="${RUN_NAME:-qwen3_4b_email_multiclass_eagle3_classify_refill}"
export GPU_ID="${GPU_ID:-3}"
export DISTILL_ARGS="${DISTILL_ARGS:-\
--model-path ${TARGET_MODEL:-${ROOT_DIR}/models/Qwen3-4B-email-multiclass-v2} \
--output-dir ${DISTILL_OUTPUT_DIR:-${ROOT_DIR}/data/finetune/eagle3_multiclass_v2_mixed} \
--input-dir ${SOURCE_DIR:-data/processed/spam_benchmark} \
--input-dir ${SOURCE_DIR_2:-data/processed/phishing_benchmark} \
--input-dir ${SOURCE_DIR_3:-data/processed/maildir_benchmark} \
--task classify_email \
--retry-rejected-tasks classify_email \
--allow-config-change \
--train-per-task ${TRAIN_PER_TASK:-2500} \
--validation-per-task ${VALIDATION_PER_TASK:-250} \
--oversample-factor ${OVERSAMPLE_FACTOR:-5} \
--training-max-length ${TRAINING_MAX_LENGTH:-768} \
--max-model-len ${MAX_MODEL_LEN:-2048} \
--max-body-chars ${MAX_BODY_CHARS:-600} \
--batch-size ${DISTILL_BATCH_SIZE:-32} \
--gpu-memory-utilization ${GPU_MEMORY_UTILIZATION:-0.75}}"

exec "${ROOT_DIR}/scripts/start_eagle3_distillation_detached.sh"
