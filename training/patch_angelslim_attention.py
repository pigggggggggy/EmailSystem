#!/usr/bin/env python3
"""Make AngelSlim target-model attention backend configurable."""

from __future__ import annotations

import argparse
from pathlib import Path

TARGET_FILE = Path("angelslim/compressor/speculative/train/models/target/target_model_wrapper.py")
ORIGINAL = "\"attn_implementation\": \"flash_attention_2\","
PATCHED = ("\"attn_implementation\": os.environ.get("
           "\"ANGELSLIM_TARGET_ATTN_IMPLEMENTATION\", \"sdpa\"),")


def patch_source(source: str) -> tuple[str, bool]:
    if PATCHED in source:
        return source, False
    if ORIGINAL not in source:
        raise ValueError("expected AngelSlim attention configuration was not found")
    if "import os" not in source:
        raise ValueError("AngelSlim target wrapper does not import os")
    return source.replace(ORIGINAL, PATCHED, 1), True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--angelslim-dir", type=Path, required=True)
    args = parser.parse_args()
    target = args.angelslim_dir / TARGET_FILE
    source = target.read_text(encoding="utf-8")
    patched, changed = patch_source(source)
    if changed:
        target.write_text(patched, encoding="utf-8")
        print(f"Patched AngelSlim target attention backend: {target}")
    else:
        print(f"AngelSlim attention patch already applied: {target}")


if __name__ == "__main__":
    main()
