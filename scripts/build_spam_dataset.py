#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from itertools import chain, islice
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from email_system.evaluation.spam_dataset import iter_enron_rows, iter_trec_rows, split_records, write_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic Enron/TREC spam benchmark splits.")
    parser.add_argument("--enron-csv", default="datasets/enron_spam_data/enron_spam_data.csv")
    parser.add_argument("--trec-root", default="datasets/trec06c")
    parser.add_argument("--output-dir", default="data/processed/spam_benchmark")
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--limit-per-source", type=int, default=None, help="Smoke-test limit; omit for the full dataset")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    enron = iter_enron_rows(args.enron_csv)
    trec = iter_trec_rows(args.trec_root)
    if args.limit_per_source is not None:
        enron = islice(enron, args.limit_per_source)
        trec = islice(trec, args.limit_per_source)
    splits, manifest = split_records(
        chain(enron, trec),
        seed=args.seed,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
    )
    manifest["inputs"] = {
        "enron_csv": str(Path(args.enron_csv)),
        "trec_root": str(Path(args.trec_root)),
        "limit_per_source": args.limit_per_source,
    }
    write_dataset(args.output_dir, splits, manifest)
    print(json.dumps({"output_dir": args.output_dir, **manifest}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
