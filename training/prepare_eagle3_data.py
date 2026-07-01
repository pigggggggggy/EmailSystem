#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare AngelSlim EAGLE3 conversation JSONL data.")
    parser.add_argument("--train-input", default="data/finetune/classification_lora/train.jsonl")
    parser.add_argument("--validation-input", default="data/finetune/classification_lora/validation.jsonl")
    parser.add_argument("--output-dir", default="data/finetune/eagle3_classification")
    parser.add_argument("--train-limit", type=int, default=5000, help="0 keeps the complete train split")
    parser.add_argument("--validation-limit", type=int, default=500, help="0 keeps the complete validation split")
    parser.add_argument("--seed", type=int, default=20260701)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train = prepare_rows(_read_jsonl(args.train_input), limit=args.train_limit, seed=args.seed)
    validation = prepare_rows(_read_jsonl(args.validation_input), limit=args.validation_limit, seed=args.seed + 1)
    _write_jsonl(output_dir / "train.jsonl", train)
    _write_jsonl(output_dir / "validation.jsonl", validation)
    manifest = {
        "task": "eagle3_draft_training",
        "format": "angelslim_conversations_v1",
        "train_input": args.train_input,
        "validation_input": args.validation_input,
        "output_dir": str(output_dir),
        "seed": args.seed,
        "train_limit": args.train_limit,
        "validation_limit": args.validation_limit,
        "train_records": len(train),
        "validation_records": len(validation),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def prepare_rows(rows: Iterable[dict], *, limit: int, seed: int) -> list[dict]:
    prepared = []
    seen_ids = set()
    for index, row in enumerate(rows):
        row_id = str(row.get("email_id") or f"row-{index}")
        if row_id in seen_ids:
            continue
        prepared.append({"id": row_id, "conversations": _validated_messages(row.get("messages"), row_id)})
        seen_ids.add(row_id)
    prepared.sort(key=lambda item: _stable_key(seed, item["id"]))
    return prepared[:limit] if limit > 0 else prepared


def _validated_messages(value: object, row_id: str) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"missing messages for {row_id}")
    messages = []
    for message in value:
        if not isinstance(message, dict):
            raise ValueError(f"invalid message for {row_id}: {message!r}")
        role = str(message.get("role", ""))
        content = str(message.get("content", ""))
        if role not in {"system", "user", "assistant"} or not content:
            raise ValueError(f"invalid message for {row_id}: role={role!r}")
        messages.append({"role": role, "content": content})
    if messages[-1]["role"] != "assistant" or not any(item["role"] == "user" for item in messages):
        raise ValueError(f"invalid conversation turns: {row_id}")
    return messages


def _stable_key(seed: int, row_id: str) -> str:
    return hashlib.sha256(f"{seed}:{row_id}".encode("utf-8")).hexdigest()


def _read_jsonl(path: str | Path) -> Iterable[dict]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {source}:{line_number}: {exc}") from exc


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
