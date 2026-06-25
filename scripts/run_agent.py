#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from email_system.agent import EmailAgentWorkflow
from email_system.io import read_emails, write_jsonl
from email_system.models import MockLLMClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the email agent workflow on a JSONL file.")
    parser.add_argument("--input", default="data/eval_sets/sample_emails.jsonl")
    parser.add_argument("--output", default="outputs/predictions/sample_predictions.jsonl")
    parser.add_argument("--backend", default="mock", choices=["mock"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    llm = MockLLMClient()
    workflow = EmailAgentWorkflow(llm)
    predictions = [workflow.run(email).to_dict() for email in read_emails(args.input)]
    write_jsonl(args.output, predictions)
    print(f"processed={len(predictions)} output={args.output}")


if __name__ == "__main__":
    main()
