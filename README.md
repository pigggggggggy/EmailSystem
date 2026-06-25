# EmailSystem

Use Qwen3-4B to build a full-stack email processing agent.

See [docs/architecture.md](docs/architecture.md) for the initial system structure, including agent workflow, skills, MCP, finetuning, and evaluation design.

## Current Skeleton

The repository now contains a minimal workflow skeleton:

- email dataclass schemas
- deterministic mock LLM client
- local transformers backend for `models/Qwen3-4B`
- classify, summarize, action-item, and reply-draft skills
- sequential agent workflow
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

## Local Qwen3-4B

The local model is expected at:

```text
models/Qwen3-4B
```

Install optional local model dependencies when needed:

```bash
pip install -e '.[qwen]'
```

Run the agent with the local Qwen3-4B transformers backend:

```bash
python scripts/run_agent.py \
  --backend transformers \
  --model-path models/Qwen3-4B \
  --input data/eval_sets/sample_emails.jsonl \
  --output outputs/predictions/qwen3_4b_predictions.jsonl
```

Run evaluation with the local model:

```bash
python scripts/run_eval.py \
  --backend transformers \
  --model-path models/Qwen3-4B \
  --input data/eval_sets/sample_emails.jsonl
```

The `models/` and `outputs/` directories are ignored by git, so model weights and run artifacts stay out of source control.
