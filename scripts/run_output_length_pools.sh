#!/usr/bin/env bash
set -euo pipefail

# Runs short- and long-output benchmark pools concurrently on separate TP1 GPUs.
# This is an evaluation utility, not an HTTP load balancer for production traffic.
MODEL_PATH="${MODEL_PATH:-models/Qwen3-4B-email-multitask-v3-maildir}"
INPUT="${INPUT:-data/finetune/multiclass_consensus_v3_maildir/validation.jsonl}"
RUN_ROOT="${RUN_ROOT:-outputs/runs/$(date -u +%Y%m%d_%H%M%S)_output_pools}"
SHORT_GPU="${SHORT_GPU:-0}"
LONG_GPU="${LONG_GPU:-1}"
SPEED_LIMIT="${SPEED_LIMIT:-200}"
QUALITY_LIMIT="${QUALITY_LIMIT:-1000}"
COMMON_ARGS=(
  --model-path "$MODEL_PATH"
  --input "$INPUT"
  --quality-mode multiclass
  --quality-limit "$QUALITY_LIMIT"
  --speed-limit "$SPEED_LIMIT"
  --continuous-batching
  --use-compiled-graphs
  --tensor-parallel-size 1
)

mkdir -p "$RUN_ROOT"
CUDA_VISIBLE_DEVICES="$SHORT_GPU" python scripts/run_parallel_eval.py \
  "${COMMON_ARGS[@]}" \
  --speed-tasks classify_email extract_action_items \
  --max-model-len 2048 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 128 \
  --run-dir "$RUN_ROOT/short_pool" >"$RUN_ROOT/short_pool.log" 2>&1 &
short_pid=$!

CUDA_VISIBLE_DEVICES="$LONG_GPU" python scripts/run_parallel_eval.py \
  "${COMMON_ARGS[@]}" \
  --skip-quality \
  --speed-tasks summarize_email draft_reply \
  --max-model-len 2048 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 64 \
  --run-dir "$RUN_ROOT/long_pool" >"$RUN_ROOT/long_pool.log" 2>&1 &
long_pid=$!

wait "$short_pid"
wait "$long_pid"
printf 'Output-pool benchmarks completed: %s\n' "$RUN_ROOT"
