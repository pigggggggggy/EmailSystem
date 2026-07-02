#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export RUN_NAME="${RUN_NAME:-qwen3_4b_email_eagle3_mixed}"
export TRAIN_DATA="${TRAIN_DATA:-${ROOT_DIR}/data/finetune/eagle3_mixed/train.jsonl}"
export EVAL_DATA="${EVAL_DATA:-${ROOT_DIR}/data/finetune/eagle3_mixed/validation.jsonl}"
export OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/eagle3/qwen3_4b_email_mixed}"
exec "${ROOT_DIR}/scripts/start_eagle3_training_detached.sh"
