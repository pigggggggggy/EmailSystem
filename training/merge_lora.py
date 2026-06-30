#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into a standalone causal-LM model directory.")
    parser.add_argument("--base-model", default="models/Qwen3-4B")
    parser.add_argument("--adapter", default="outputs/lora/qwen3_4b_classification_lora")
    parser.add_argument("--output-dir", default="models/Qwen3-4B-email-classifier")
    parser.add_argument("--safe-serialization", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not Path(args.adapter).exists():
        raise SystemExit(f"Adapter directory not found: {args.adapter}")

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype="auto",
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    merged = model.merge_and_unload()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(output_dir, safe_serialization=args.safe_serialization)
    tokenizer.save_pretrained(output_dir)
    print(f"Merged model written to {output_dir}")


if __name__ == "__main__":
    main()
