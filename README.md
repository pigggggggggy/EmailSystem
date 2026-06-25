# EmailSystem

Use Qwen3-4B to build a full-stack email processing agent.

See [docs/architecture.md](docs/architecture.md) for the initial system structure, including agent workflow, skills, MCP, finetuning, and evaluation design.

## Current Skeleton

The repository now contains a minimal, GPU-free workflow skeleton:

- email dataclass schemas
- deterministic mock LLM client
- classify, summarize, action-item, and reply-draft skills
- sequential agent workflow
- sample JSONL evaluation set
- classification and latency metrics
- runnable agent and eval scripts

## Quick Start

Run the agent on the sample emails:

```bash
python scripts/run_agent.py --input data/eval_sets/sample_emails.jsonl --output outputs/predictions/sample_predictions.jsonl
```

Run the evaluation loop:

```bash
python scripts/run_eval.py --input data/eval_sets/sample_emails.jsonl
```

The default backend is `mock`, so these commands do not download or run Qwen3-4B yet. The real Qwen/vLLM integration should be added behind `src/email_system/models/` without changing the workflow API.

Run unit tests:

```bash
PYTHONPATH=src python -m unittest discover -s tests/unit
```
