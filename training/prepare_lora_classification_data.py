#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from collections import Counter
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from email_system.models.chat_prompts import messages_for_task


DEFAULT_INPUT_DIRS = [
    "data/processed/spam_benchmark",
    "data/processed/phishing_benchmark",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare chat-format LoRA data for email classification.")
    parser.add_argument(
        "--input-dir",
        action="append",
        default=None,
        help="Processed benchmark directory containing train/validation JSONL. Repeat to merge datasets. Defaults to spam + phishing benchmarks.",
    )
    parser.add_argument("--output-dir", default="data/finetune/classification_lora")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--validation-split", default="validation")
    parser.add_argument("--max-body-chars", type=int, default=6000)
    parser.add_argument("--spam-confidence", type=float, default=0.95)
    parser.add_argument("--ham-confidence", type=float, default=0.90)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dirs = [Path(value) for value in (args.input_dir or DEFAULT_INPUT_DIRS)]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows, train_sources = _collect_split_rows(input_dirs, args.train_split)
    validation_rows, validation_sources = _collect_split_rows(input_dirs, args.validation_split)

    train_count = _write_chat_jsonl(
        output_dir / "train.jsonl",
        train_rows,
        max_body_chars=args.max_body_chars,
        spam_confidence=args.spam_confidence,
        ham_confidence=args.ham_confidence,
    )
    validation_count = _write_chat_jsonl(
        output_dir / "validation.jsonl",
        validation_rows,
        max_body_chars=args.max_body_chars,
        spam_confidence=args.spam_confidence,
        ham_confidence=args.ham_confidence,
    )
    manifest = {
        "task": "classification_lora",
        "source_dirs": [str(input_dir) for input_dir in input_dirs],
        "output_dir": str(output_dir),
        "max_body_chars": args.max_body_chars,
        "train_records": train_count,
        "validation_records": validation_count,
        "split_sources": {
            "train": dict(train_sources),
            "validation": dict(validation_sources),
        },
        "target_mapping": {
            "spam": {"category": "spam", "priority": "normal", "confidence": args.spam_confidence},
            "ham": {"category": "automated_email", "priority": "normal", "confidence": args.ham_confidence},
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def _collect_split_rows(input_dirs: list[Path], split: str) -> tuple[list[dict], Counter]:
    rows = []
    sources: Counter = Counter()
    for input_dir in input_dirs:
        path = input_dir / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"missing split file: {path}")
        source_key = str(input_dir)
        split_rows = list(_read_jsonl(path))
        rows.extend(split_rows)
        sources[source_key] += len(split_rows)
    return rows, sources


def _write_chat_jsonl(
    path: Path,
    rows: Iterable[dict],
    *,
    max_body_chars: int,
    spam_confidence: float,
    ham_confidence: float,
) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            label = row.get("labels", {}).get("spam_label")
            if label not in {"spam", "ham"}:
                raise ValueError(f"unsupported spam_label for {row.get('email_id')}: {label!r}")
            prompt = _email_prompt(row, max_body_chars=max_body_chars)
            messages = messages_for_task(prompt, task="classify_email")
            messages.append({"role": "assistant", "content": _target_json(label, spam_confidence, ham_confidence)})
            item = {
                "email_id": row.get("email_id"),
                "source": row.get("source"),
                "spam_label": label,
                "messages": messages,
            }
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1
    return count


def _email_prompt(row: dict, *, max_body_chars: int) -> str:
    body = str(row.get("body_text", ""))[:max_body_chars]
    to_values = row.get("to") or []
    if isinstance(to_values, str):
        to_text = to_values
    else:
        to_text = ", ".join(str(value) for value in to_values)
    return "\n".join(
        [
            f"Subject: {row.get('subject', '')}",
            f"From: {row.get('from', '')}",
            f"To: {to_text}",
            f"Timestamp: {row.get('timestamp', '')}",
            "Body:",
            body,
        ]
    )


def _target_json(label: str, spam_confidence: float, ham_confidence: float) -> str:
    if label == "spam":
        target = {"category": "spam", "priority": "normal", "confidence": spam_confidence}
    else:
        target = {"category": "automated_email", "priority": "normal", "confidence": ham_confidence}
    return json.dumps(target, ensure_ascii=False, separators=(",", ":"))


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc


if __name__ == "__main__":
    main()
