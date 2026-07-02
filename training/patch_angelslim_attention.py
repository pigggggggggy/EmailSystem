#!/usr/bin/env python3
"""Apply local compatibility fixes to the pinned AngelSlim checkout."""

from __future__ import annotations

import argparse
from pathlib import Path

ATTENTION_FILE = Path("angelslim/compressor/speculative/train/models/target/target_model_wrapper.py")
DATASET_FILE = Path("angelslim/compressor/speculative/train/data/dataset.py")
ATTENTION_ORIGINAL = "\"attn_implementation\": \"flash_attention_2\","
ATTENTION_PATCHED = ("\"attn_implementation\": os.environ.get("
                     "\"ANGELSLIM_TARGET_ATTN_IMPLEMENTATION\", \"sdpa\"),")
DATASET_ORIGINAL = "target_model_type=self.target_model_type,"
DATASET_PATCHED = ("target_model_type=(None if data_args.modal_type == \"LLM\" "
                   "else self.target_model_type),")


def patch_attention_source(source: str) -> tuple[str, bool]:
    if ATTENTION_PATCHED in source:
        return source, False
    if ATTENTION_ORIGINAL not in source:
        raise ValueError("expected AngelSlim attention configuration was not found")
    if "import os" not in source:
        raise ValueError("AngelSlim target wrapper does not import os")
    return source.replace(ATTENTION_ORIGINAL, ATTENTION_PATCHED, 1), True


def patch_dataset_source(source: str) -> tuple[str, bool]:
    if DATASET_PATCHED in source:
        return source, False
    occurrences = source.count(DATASET_ORIGINAL)
    if occurrences != 2:
        raise ValueError(f"expected two AngelSlim dataset target-type arguments, found {occurrences}")
    return source.replace(DATASET_ORIGINAL, DATASET_PATCHED), True


def patch_file(path: Path, transform, label: str) -> None:
    source = path.read_text(encoding="utf-8")
    patched, changed = transform(source)
    if changed:
        path.write_text(patched, encoding="utf-8")
        print(f"Patched AngelSlim {label}: {path}")
    else:
        print(f"AngelSlim {label} patch already applied: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--angelslim-dir", type=Path, required=True)
    args = parser.parse_args()
    patch_file(args.angelslim_dir / ATTENTION_FILE, patch_attention_source, "target attention backend")
    patch_file(args.angelslim_dir / DATASET_FILE, patch_dataset_source, "LLM dataset registration")


if __name__ == "__main__":
    main()
