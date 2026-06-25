#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from email_system.agent import EmailAgentWorkflow
from email_system.imap_mail import IMAPConfig, IMAPEmailClient
from email_system.io import write_jsonl
from email_system.models import build_llm_client


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the email agent on an IMAP mailbox.")
    parser.add_argument("--host", default="imap.gmail.com")
    parser.add_argument("--port", type=int, default=993)
    parser.add_argument("--mailbox", default="INBOX")
    parser.add_argument("--user", default=os.environ.get("EMAILSYSTEM_IMAP_USER"))
    parser.add_argument("--password-env", default="EMAILSYSTEM_IMAP_PASSWORD")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--search", default="ALL")
    parser.add_argument("--output", default="outputs/predictions/imap_predictions.jsonl")
    parser.add_argument("--backend", default="mock", choices=["mock", "transformers", "vllm"])
    parser.add_argument("--model-path", default="models/Qwen3-4B")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    password = os.environ.get(args.password_env)
    if not args.user:
        raise SystemExit("Missing --user or EMAILSYSTEM_IMAP_USER")
    if not password:
        raise SystemExit(f"Missing password environment variable: {args.password_env}")
    imap_client = IMAPEmailClient(
        user=args.user,
        password=password,
        config=IMAPConfig(host=args.host, port=args.port, mailbox=args.mailbox),
    )
    emails = imap_client.fetch_recent(limit=args.limit, search=args.search)
    llm = build_llm_client(
        args.backend,
        model_path=args.model_path,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    workflow = EmailAgentWorkflow(llm)
    predictions = [workflow.run(email).to_dict() for email in emails]
    write_jsonl(args.output, predictions)
    print(f"processed={len(predictions)} backend={args.backend} output={args.output}")


if __name__ == "__main__":
    main()
