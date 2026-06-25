#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from email_system.agent import EmailAgentWorkflow
from email_system.io import read_emails, write_jsonl
from email_system.models import build_llm_client


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the email agent workflow on a JSONL file.")
    parser.add_argument("--input", default="data/eval_sets/sample_emails.jsonl")
    parser.add_argument("--output", default="outputs/predictions/sample_predictions.jsonl")
    parser.add_argument("--backend", default="mock", choices=["mock", "transformers"])
    parser.add_argument("--model-path", default="models/Qwen3-4B")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    llm = build_llm_client(
        args.backend,
        model_path=args.model_path,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
    )
    workflow = EmailAgentWorkflow(llm)
    predictions = [workflow.run(email).to_dict() for email in read_emails(args.input)]
    write_jsonl(args.output, predictions)
    print(f"processed={len(predictions)} backend={args.backend} output={args.output}")


if __name__ == "__main__":
    main()
