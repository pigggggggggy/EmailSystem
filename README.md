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
- LangGraph-style agent workflow with execution trace
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

The agent now follows a LangGraph-style workflow inspired by the automatic email agent pattern:

```text
START
  -> read_email
  -> classify_intent
  -> bug_tracking | search_documentation
  -> write_response
  -> human_review
  -> send_reply
  -> END
```

When `langgraph` is installed, `EmailAgentWorkflow` compiles this with `StateGraph`. In minimal test environments it falls back to the same graph order locally, so mock tests do not require optional agent dependencies.

Each prediction includes `workflow_trace`, `timings_ms`, and a `memory` snapshot. Short-term memory keeps recent context by `thread_id` inside the running process. Long-term memory currently supports an in-memory store for tests and an append-only JSONL store for local persistence.

The mail interface is modeled after `dicklesworthstone/mcp_agent_mail`, which provides a FastMCP mail-like coordination layer with identities, inbox/outbox, searchable threads, and message sending. The current code uses `NoopMailMCPClient` as a safe local adapter; wire a running MCP Agent Mail server behind `MailMCPClient` when you are ready for real mailbox operations.

Install optional LangGraph/MCP workflow dependencies:

```bash
pip install -e '.[agent]'
```

## Gmail Trial

Gmail integration uses the official Gmail API OAuth desktop flow. Keep credentials outside git:

```text
secrets/gmail_credentials.json
 data/auth/gmail_token.json
```

Setup:

1. Enable the Gmail API in a Google Cloud project.
2. Configure the OAuth consent screen.
3. Create an OAuth Client ID with application type `Desktop app`.
4. Download the client JSON to `secrets/gmail_credentials.json`.
5. Install dependencies:

```bash
pip install -e '.[gmail]'
```

Import recent Gmail messages to JSONL:

```bash
python scripts/import_gmail.py \
  --credentials secrets/gmail_credentials.json \
  --token data/auth/gmail_token.json \
  --query "in:inbox newer_than:30d" \
  --limit 10 \
  --output data/eval_sets/gmail_inbox.jsonl
```

Run the agent directly on Gmail in safe dry-run mode. This reads messages and generates replies, but does not send or create drafts:

```bash
python scripts/run_gmail_agent.py \
  --backend mock \
  --send-mode dry_run \
  --query "in:inbox newer_than:30d" \
  --limit 3 \
  --output outputs/predictions/gmail_predictions.jsonl
```

Create Gmail drafts instead of sending:

```bash
python scripts/run_gmail_agent.py \
  --backend vllm \
  --model-path models/Qwen3-4B \
  --send-mode draft \
  --query "in:inbox newer_than:30d" \
  --limit 3
```

Avoid `--send-mode send` until the workflow has been reviewed on your own mailbox; it sends real email.

## Gmail Over IMAP

If Gmail API/OAuth is too heavy, use Gmail IMAP with an App Password. This path does not require Google Cloud.

Set credentials in your shell. Do not commit them:

```bash
export EMAILSYSTEM_IMAP_USER="your-address@gmail.com"
export EMAILSYSTEM_IMAP_PASSWORD="your-16-character-app-password-without-spaces"
```

Import recent Gmail messages over IMAP:

```bash
python scripts/import_imap.py \
  --host imap.gmail.com \
  --user "$EMAILSYSTEM_IMAP_USER" \
  --password-env EMAILSYSTEM_IMAP_PASSWORD \
  --mailbox INBOX \
  --limit 10 \
  --output data/eval_sets/imap_inbox.jsonl
```

Run the real LangGraph agent directly on Gmail IMAP with local Qwen3-4B through vLLM:

```bash
python scripts/run_imap_agent.py \
  --backend vllm \
  --model-path models/Qwen3-4B \
  --host imap.gmail.com \
  --user "$EMAILSYSTEM_IMAP_USER" \
  --password-env EMAILSYSTEM_IMAP_PASSWORD \
  --mailbox INBOX \
  --limit 3 \
  --output outputs/predictions/imap_predictions.jsonl
```

The command requires LangGraph and records the actual graph backend, model backend, conditional route, and delivery decision in every output row. This IMAP path remains read-only: the `send_reply` node runs, but it does not send or draft replies.

## Build the spam benchmark

Normalize and deterministically split the local Enron Spam and TREC06c datasets:

```bash
python scripts/build_spam_dataset.py \
  --output-dir data/processed/spam_benchmark
```

For a quick parser check, add `--limit-per-source 20`. Generated JSONL files are local artifacts and are ignored by Git; `manifest.json` records the seed, split sizes, label counts, duplicate removal, and source counts.

## Run the independent benchmark

Measure classification quality only, while reporting speed independently for classification, summarization, action extraction, and reply drafting:

```bash
python scripts/run_independent_eval.py \
  --backend vllm \
  --model-path models/Qwen3-4B \
  --input data/processed/spam_benchmark/test.jsonl \
  --quality-limit 1000 \
  --speed-limit 100
```

Omit `--quality-limit` for all 12,807 test emails. The default `--max-body-chars 12000` keeps long public-dataset messages within a stable input budget. Results include classification predictions, per-request speed samples, metrics, configuration, and a Markdown report under `outputs/runs/`.
