#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ANGELSLIM_DIR="${ANGELSLIM_DIR:-${ROOT_DIR}/third_party/AngelSlim}"
TARGET_MODEL="${TARGET_MODEL:-${ROOT_DIR}/models/Qwen3-4B-email-classifier-ckpt1563}"
DRAFT_CONFIG="${DRAFT_CONFIG:-${ROOT_DIR}/training/configs/qwen3_4b_email_eagle3.json}"
TRAIN_DATA="${TRAIN_DATA:-${ROOT_DIR}/data/finetune/eagle3_classification/train.jsonl}"
EVAL_DATA="${EVAL_DATA:-${ROOT_DIR}/data/finetune/eagle3_classification/validation.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/eagle3/qwen3_4b_email_classifier}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${ROOT_DIR}/training/configs/deepspeed_eagle3_zero3.json}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
MASTER_PORT="${MASTER_PORT:-29610}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-2048}"
EPOCHS="${EPOCHS:-3}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
NUM_PROC="${NUM_PROC:-8}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
DRY_RUN="${DRY_RUN:-0}"
for required in "$ANGELSLIM_DIR/tools/train_eagle3_online.py" "$TARGET_MODEL/config.json" "$DRAFT_CONFIG" "$TRAIN_DATA" "$EVAL_DATA" "$DEEPSPEED_CONFIG"; do
  [[ -e "$required" ]] || { echo "Required file is missing: $required" >&2; exit 1; }
done
IFS=',' read -r -a gpu_array <<< "$GPU_IDS"
NPROC_PER_NODE="${#gpu_array[@]}"
mkdir -p "$OUTPUT_DIR"
cd "$ANGELSLIM_DIR"
command=(torchrun --nproc_per_node "$NPROC_PER_NODE" --master_port "$MASTER_PORT" tools/train_eagle3_online.py
  --target_model_name_or_path "$TARGET_MODEL" --draft_model_config_path "$DRAFT_CONFIG"
  --train_data_path "$TRAIN_DATA" --eval_data_path "$EVAL_DATA" --output_dir "$OUTPUT_DIR"
  --num_train_epochs "$EPOCHS" --per_device_train_batch_size "$TRAIN_BATCH_SIZE"
  --per_device_eval_batch_size "$EVAL_BATCH_SIZE" --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
  --learning_rate "$LEARNING_RATE" --weight_decay 0.0 --warmup_ratio 0.1 --lr_scheduler_type constant
  --logging_steps 10 --save_strategy epoch --eval_strategy epoch --save_total_limit "$SAVE_TOTAL_LIMIT"
  --model_max_length "$MODEL_MAX_LENGTH" --chat_template_type qwen3 --num_proc "$NUM_PROC"
  --deepspeed "$DEEPSPEED_CONFIG" --bf16 --report_to none --run_name qwen3-4b-email-eagle3)
if [[ -n "$EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2206
  extra_args_array=($EXTRA_ARGS)
  command+=("${extra_args_array[@]}")
fi
printf 'Command: '; printf '%q ' "${command[@]}"; echo
if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi
CUDA_VISIBLE_DEVICES="$GPU_IDS" "${command[@]}"
