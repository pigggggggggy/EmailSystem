#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 6 ]; then
  echo "Usage: $0 PYTHON MODEL_PATH INPUT RUN_ROOT NGRAM_MIN NGRAM_MAX [run_parallel_eval options...]" >&2
  exit 2
fi

PYTHON_BIN="$1"
MODEL_PATH="$2"
INPUT_PATH="$3"
RUN_ROOT="$4"
NGRAM_MIN="$5"
NGRAM_MAX="$6"
shift 6

mkdir -p "$RUN_ROOT"

"$PYTHON_BIN" scripts/run_parallel_eval.py   --model-path "$MODEL_PATH"   --input "$INPUT_PATH"   --run-dir "$RUN_ROOT/baseline_short"   --speed-tasks classify_email summarize_email   "$@"

"$PYTHON_BIN" scripts/run_parallel_eval.py   --model-path "$MODEL_PATH"   --input "$INPUT_PATH"   --run-dir "$RUN_ROOT/ngram_action_items_st4"   --skip-quality   --speed-tasks extract_action_items   --ngram-prompt-lookup-min "$NGRAM_MIN"   --ngram-prompt-lookup-max "$NGRAM_MAX"   --speculative-tokens 5   "$@"

"$PYTHON_BIN" scripts/run_parallel_eval.py   --model-path "$MODEL_PATH"   --input "$INPUT_PATH"   --run-dir "$RUN_ROOT/ngram_draft_reply_st6"   --skip-quality   --speed-tasks draft_reply   --ngram-prompt-lookup-min "$NGRAM_MIN"   --ngram-prompt-lookup-max "$NGRAM_MAX"   --speculative-tokens 6   "$@"

"$PYTHON_BIN" scripts/merge_task_routed_eval.py --run-root "$RUN_ROOT"

echo "Completed task-routed runs under $RUN_ROOT (combined report: $RUN_ROOT/combined/report.md)"
