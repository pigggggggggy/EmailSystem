#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export RUN_NAME="${RUN_NAME:-qwen3_4b_email_multiclass_eagle3}"
export TARGET_MODEL="${TARGET_MODEL:-${ROOT_DIR}/models/Qwen3-4B-email-multiclass-v2}"
export TRAIN_DATA="${TRAIN_DATA:-${ROOT_DIR}/data/finetune/eagle3_multiclass_v2_mixed/train.jsonl}"
export EVAL_DATA="${EVAL_DATA:-${ROOT_DIR}/data/finetune/eagle3_multiclass_v2_mixed/validation.jsonl}"
export OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/eagle3/qwen3_4b_email_multiclass_v2}"
export GPU_IDS="${GPU_IDS:-0,1,2,3}"
export MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-512}"
export EPOCHS="${EPOCHS:-3}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
export TARGET_ATTN_IMPLEMENTATION="${TARGET_ATTN_IMPLEMENTATION:-sdpa}"

exec "${ROOT_DIR}/scripts/start_eagle3_training_detached.sh"
