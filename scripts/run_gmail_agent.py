#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from email_system.agent import EmailAgentWorkflow
from email_system.gmail import GMAIL_COMPOSE_SCOPE, GMAIL_READONLY_SCOPE, GMAIL_SEND_SCOPE, GmailAPIClient, build_gmail_service
from email_system.io import write_jsonl
from email_system.mcp import GmailMailMCPClient
from email_system.models import build_llm_client


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the email agent directly on recent Gmail messages.")
    parser.add_argument("--credentials", default="secrets/gmail_credentials.json")
    parser.add_argument("--token", default="data/auth/gmail_token.json")
    parser.add_argument("--query", default="in:inbox newer_than:30d")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--output", default="outputs/predictions/gmail_predictions.jsonl")
    parser.add_argument("--send-mode", default="dry_run", choices=["dry_run", "draft", "send"])
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
    scopes = [GMAIL_READONLY_SCOPE]
    if args.send_mode == "draft":
        scopes.append(GMAIL_COMPOSE_SCOPE)
    elif args.send_mode == "send":
        scopes.append(GMAIL_SEND_SCOPE)
    service = build_gmail_service(credentials_path=args.credentials, token_path=args.token, scopes=scopes)
    gmail_client = GmailAPIClient(service)
    emails = gmail_client.list_emails(query=args.query, limit=args.limit)
    llm = build_llm_client(
        args.backend,
        model_path=args.model_path,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    workflow = EmailAgentWorkflow(llm, mail_client=GmailMailMCPClient(gmail_client, send_mode=args.send_mode))
    predictions = [workflow.run(email).to_dict() for email in emails]
    write_jsonl(args.output, predictions)
    print(f"processed={len(predictions)} send_mode={args.send_mode} backend={args.backend} output={args.output}")


if __name__ == "__main__":
    main()
