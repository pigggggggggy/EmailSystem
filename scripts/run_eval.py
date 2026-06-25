#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from email_system.agent import EmailAgentWorkflow
from email_system.evaluation import evaluate_predictions
from email_system.io import read_emails, read_jsonl, write_jsonl
from email_system.models import MockLLMClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run agent evaluation.")
    parser.add_argument("--input", default="data/eval_sets/sample_emails.jsonl")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--backend", default="mock", choices=["mock"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else Path("outputs/runs") / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_mock")
    run_dir.mkdir(parents=True, exist_ok=True)

    workflow = EmailAgentWorkflow(MockLLMClient())
    emails = read_emails(args.input)
    predictions = [workflow.run(email).to_dict() for email in emails]
    gold_rows = read_jsonl(args.input)
    result = evaluate_predictions(gold_rows, predictions)

    write_jsonl(run_dir / "predictions.jsonl", predictions)
    (run_dir / "metrics.json").write_text(json.dumps(result.metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "report.md").write_text(render_report(result.metrics), encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "metrics": result.metrics}, ensure_ascii=False, indent=2))


def render_report(metrics: dict) -> str:
    classification = metrics["classification"]
    latency = metrics["latency"]
    return "\n".join(
        [
            "# EmailSystem Evaluation Report",
            "",
            "## Classification",
            f"- Accuracy: {classification['accuracy']:.3f}",
            f"- Macro F1: {classification['macro_f1']:.3f}",
            "",
            "## Latency",
            f"- Emails: {latency['emails']}",
            f"- Avg end-to-end ms: {latency['avg_end_to_end_ms']:.2f}",
            "",
        ]
    )


if __name__ == "__main__":
    main()
