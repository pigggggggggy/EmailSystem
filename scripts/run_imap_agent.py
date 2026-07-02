#!/usr/bin/env python3
from __future__ import annotations

import argparse
import imaplib
import importlib.util
import os
import socket
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
    parser.add_argument("--timeout", type=float, default=20.0, help="IMAP connection timeout in seconds")
    parser.add_argument("--user", default=os.environ.get("EMAILSYSTEM_IMAP_USER"))
    parser.add_argument("--password-env", default="EMAILSYSTEM_IMAP_PASSWORD")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--search", default="ALL")
    parser.add_argument("--output", default="outputs/predictions/imap_predictions.jsonl")
    parser.add_argument("--backend", default="vllm", choices=["mock", "transformers", "vllm"])
    parser.add_argument("--model-path", default="models/Qwen3-4B")
    parser.add_argument("--eagle3-model-path", default=None)
    parser.add_argument("--speculative-tokens", type=int, default=3)
    parser.add_argument(
        "--allow-fallback-graph",
        action="store_true",
        help="Allow local sequential fallback when LangGraph is not installed",
    )
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
    if not args.allow_fallback_graph and importlib.util.find_spec("langgraph") is None:
        raise SystemExit(
            "LangGraph is not installed. Install it with: pip install -e '.[agent]' "
            "or explicitly pass --allow-fallback-graph."
        )
    if args.backend != "mock" and not Path(args.model_path).exists():
        raise SystemExit(f"Model path does not exist: {args.model_path}")
    print(f"[1/4] Connecting to {args.host}:{args.port} as {args.user}...", flush=True)
    imap_client = IMAPEmailClient(
        user=args.user,
        password=password,
        config=IMAPConfig(
            host=args.host,
            port=args.port,
            mailbox=args.mailbox,
            timeout=args.timeout,
        ),
    )
    try:
        emails = imap_client.fetch_recent(limit=args.limit, search=args.search)
    except imaplib.IMAP4.error as exc:
        raise SystemExit(
            "IMAP authentication or mailbox operation failed. Check the Gmail address, "
            "use a newly generated App Password, and confirm IMAP access is allowed. "
            f"Server response: {exc}"
        ) from exc
    except (TimeoutError, socket.timeout) as exc:
        raise SystemExit(
            f"IMAP connection timed out after {args.timeout:g}s. Check network access to "
            f"{args.host}:{args.port} from this container."
        ) from exc
    except OSError as exc:
        raise SystemExit(f"IMAP connection failed: {exc}") from exc
    print(f"[2/4] Fetched {len(emails)} email(s) from {args.mailbox}.", flush=True)
    print(f"[3/4] Loading {args.backend} backend and running workflow...", flush=True)
    llm = build_llm_client(
        args.backend,
        model_path=args.model_path,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        speculative_model_path=args.eagle3_model_path,
        speculative_tokens=args.speculative_tokens,
    )
    workflow = EmailAgentWorkflow(llm)
    if not args.allow_fallback_graph and workflow.graph_backend != "langgraph":
        raise SystemExit("The workflow did not compile with LangGraph.")
    print(f"      graph={workflow.graph_backend} model={type(llm).__name__}", flush=True)
    predictions = [workflow.run(email).to_dict() for email in emails]
    write_jsonl(args.output, predictions)
    print(f"[4/4] processed={len(predictions)} backend={args.backend} output={args.output}", flush=True)


if __name__ == "__main__":
    main()
