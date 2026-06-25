# EmailSystem

Use Qwen3-4B to build a full-stack email processing agent.

See [docs/architecture.md](docs/architecture.md) for the initial system structure, including agent workflow, skills, MCP, finetuning, and evaluation design.

## Current Skeleton

The repository now contains a minimal workflow skeleton:

- email dataclass schemas
- deterministic mock LLM client
- local vLLM backend for `models/Qwen3-4B`
- local transformers backend for debugging fallback
- classify, summarize, action-item, and reply-draft skills
- node-based agent workflow with execution trace
- short-term thread memory and local long-term memory stores
- sample JSONL evaluation set
- classification and latency metrics
- runnable agent and eval scripts

## Quick Start

Run the agent on the sample emails with the mock backend:

```bash
python scripts/run_agent.py --input data/eval_sets/sample_emails.jsonl --output outputs/predictions/sample_predictions.jsonl
```

Run the evaluation loop with the mock backend:

```bash
python scripts/run_eval.py --input data/eval_sets/sample_emails.jsonl
```

Run unit tests:

```bash
PYTHONPATH=src python -m unittest discover -s tests/unit
```

## Local Qwen3-4B With vLLM

The local model is expected at:

```text
models/Qwen3-4B
```

Install optional vLLM dependencies when needed:

```bash
pip install -e '.[vllm]'
```

Run the agent with the local Qwen3-4B vLLM backend:

```bash
python scripts/run_agent.py \
  --backend vllm \
  --model-path models/Qwen3-4B \
  --max-model-len 8192 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9 \
  --input data/eval_sets/sample_emails.jsonl \
  --output outputs/predictions/qwen3_4b_vllm_predictions.jsonl
```

Run evaluation with vLLM:

```bash
python scripts/run_eval.py \
  --backend vllm \
  --model-path models/Qwen3-4B \
  --max-model-len 8192 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9 \
  --input data/eval_sets/sample_emails.jsonl
```

For low-level debugging without vLLM, the `transformers` backend is still available:

```bash
pip install -e '.[qwen]'
python scripts/run_agent.py --backend transformers --model-path models/Qwen3-4B
```

The `models/` and `outputs/` directories are ignored by git, so model weights and run artifacts stay out of source control.

## Troubleshooting vLLM JSON Output

If Qwen3 returns empty text or non-JSON text for classification, the workflow now falls back to safe defaults and records the issue in `skill_errors`. The vLLM and transformers clients try to disable Qwen3 thinking mode with `enable_thinking=False` when the tokenizer supports it.

Useful checks:

```bash
python -c "import vllm; print(vllm.__version__)"
nvidia-smi
```

If parse errors are high, try increasing task `max_tokens`, reducing prompt size, or inspecting `outputs/runs/<run>/predictions.jsonl` for `skill_errors`.

## Workflow And Memory

The agent now runs as a node-based workflow:

```text
load_memory
  -> classify_email
  -> summarize_email
  -> extract_action_items
  -> draft_reply
  -> human_review_policy
  -> save_memory
```

Each prediction includes `workflow_trace`, `timings_ms`, and a `memory` snapshot. Short-term memory keeps recent context by `thread_id` inside the running process. Long-term memory currently supports an in-memory store for tests and an append-only JSONL store for local persistence.
