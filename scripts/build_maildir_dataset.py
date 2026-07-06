#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from itertools import islice
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from email_system.evaluation.spam_dataset import iter_maildir_rows, split_records, write_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic Enron Maildir benchmark splits.")
    parser.add_argument("--maildir", default="datasets/maildir")
    parser.add_argument("--output-dir", default="data/processed/maildir_benchmark")
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=None, help="Smoke-test limit; omit for the full dataset")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = iter_maildir_rows(args.maildir)
    if args.limit is not None:
        rows = islice(rows, args.limit)
    splits, manifest = split_records(
        rows,
        seed=args.seed,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
    )
    manifest["task"] = "unlabeled_email_classification_candidates"
    manifest["inputs"] = {
        "maildir": str(Path(args.maildir)),
        "limit": args.limit,
    }
    write_dataset(args.output_dir, splits, manifest)
    print(json.dumps({"output_dir": args.output_dir, **manifest}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
