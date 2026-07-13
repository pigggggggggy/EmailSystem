#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantize a causal LM with AutoAWQ for vLLM evaluation.")
    parser.add_argument("--model-path", default="models/Qwen3-4B-email-multitask-v3-maildir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bits", type=int, default=4, choices=[4, 8], help="AWQ weight bits. 4-bit is the standard AWQ path; 8-bit depends on AutoAWQ support.")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--version", default="GEMM", choices=["GEMM", "GEMV"])
    parser.add_argument("--zero-point", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--calib-data", default=None, help="Optional JSONL email dataset for calibration text.")
    parser.add_argument("--calib-limit", type=int, default=512)
    parser.add_argument("--max-body-chars", type=int, default=2000)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "AutoAWQ is not installed in this Python environment. Install it in the vllm env first, for example:\n"
            "  /opt/conda/envs/vllm/bin/pip install autoawq\n"
            "Then rerun this script with /opt/conda/envs/vllm/bin/python."
        ) from exc

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    model = AutoAWQForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
        safetensors=True,
    )
    quant_config = {
        "zero_point": args.zero_point,
        "q_group_size": args.group_size,
        "w_bit": args.bits,
        "version": args.version,
    }
    calib_data = load_calibration_texts(args.calib_data, limit=args.calib_limit, max_body_chars=args.max_body_chars)
    model.quantize(tokenizer, quant_config=quant_config, calib_data=calib_data if calib_data else "pileval")
    model.save_quantized(str(output_dir), safetensors=True)
    tokenizer.save_pretrained(output_dir)
    manifest = {
        "task": "awq_quantization",
        "base_model": args.model_path,
        "output_dir": str(output_dir),
        "bits": args.bits,
        "group_size": args.group_size,
        "version": args.version,
        "zero_point": args.zero_point,
        "calib_data": args.calib_data,
        "calib_limit": args.calib_limit,
        "max_body_chars": args.max_body_chars,
        "note": "4-bit AWQ is the standard path. 8-bit AWQ support depends on the installed AutoAWQ/vLLM versions.",
    }
    (output_dir / "quantization_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


def load_calibration_texts(path: str | None, *, limit: int, max_body_chars: int) -> list[str]:
    if not path:
        return []
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append(email_prompt(row, max_body_chars=max_body_chars))
            if len(rows) >= limit:
                break
    return rows


def email_prompt(row: dict, *, max_body_chars: int) -> str:
    to_values = row.get("to") or []
    if isinstance(to_values, list):
        to_text = ", ".join(str(value) for value in to_values)
    else:
        to_text = str(to_values)
    sender = row.get("from") or row.get("sender") or ""
    return "\n".join(
        [
            f"Subject: {row.get('subject', '')}",
            f"From: {sender}",
            f"To: {to_text}",
            f"Timestamp: {row.get('timestamp', '')}",
            "Body:",
            str(row.get("body_text", ""))[:max_body_chars],
        ]
    )


if __name__ == "__main__":
    main()
