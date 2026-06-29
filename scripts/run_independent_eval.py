#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from email_system.evaluation.independent_benchmark import (
    TASKS,
    run_classification_quality,
    run_task_speed,
    select_benchmark_rows,
    truncate_body_text,
)
from email_system.io import read_jsonl, write_jsonl
from email_system.models import build_llm_client


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Independent classification-quality and per-task speed benchmark.")
    parser.add_argument("--input", default="data/processed/spam_benchmark/test.jsonl")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--backend", default="vllm", choices=["mock", "transformers", "vllm"])
    parser.add_argument("--model-path", default="models/Qwen3-4B")
    parser.add_argument("--quality-limit", type=int, default=None)
    parser.add_argument("--max-body-chars", type=int, default=12000)
    parser.add_argument("--speed-limit", type=int, default=100)
    parser.add_argument("--speed-warmup", type=int, default=2)
    parser.add_argument("--speed-tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--skip-quality", action="store_true")
    parser.add_argument("--skip-speed", action="store_true")
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.skip_quality and args.skip_speed:
        raise SystemExit("Both quality and speed phases are disabled.")
    rows = truncate_body_text(read_jsonl(args.input), args.max_body_chars)
    quality_rows = select_benchmark_rows(rows, args.quality_limit, seed=args.seed)
    speed_rows = select_benchmark_rows(rows, args.speed_limit, seed=args.seed)
    run_dir = Path(args.run_dir) if args.run_dir else Path("outputs/runs") / datetime.now(timezone.utc).strftime(
        f"%Y%m%d_%H%M%S_{args.backend}_independent"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading backend={args.backend} model={args.model_path}", flush=True)
    llm = build_llm_client(
        args.backend,
        model_path=args.model_path,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    metrics = {}
    if not args.skip_quality:
        print(f"Running classification quality on {len(quality_rows)} emails...", flush=True)
        predictions, quality_metrics = run_classification_quality(
            llm,
            quality_rows,
            progress_callback=lambda current, total: _print_progress("quality", current, total),
        )
        write_jsonl(run_dir / "classification_predictions.jsonl", predictions)
        metrics["quality"] = {"classification": quality_metrics}

    if not args.skip_speed:
        print(f"Running per-task speed on {len(speed_rows)} emails x {len(args.speed_tasks)} tasks...", flush=True)
        samples, speed_metrics = run_task_speed(
            llm,
            speed_rows,
            tasks=args.speed_tasks,
            warmup=args.speed_warmup,
        )
        write_jsonl(run_dir / "speed_samples.jsonl", samples)
        metrics["speed"] = speed_metrics

    config = vars(args).copy()
    config["input_records"] = len(rows)
    (run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (run_dir / "report.md").write_text(render_report(metrics), encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "metrics": metrics}, ensure_ascii=False, indent=2))


def _print_progress(phase: str, current: int, total: int) -> None:
    if current == total or current % 100 == 0:
        print(f"{phase}: {current}/{total}", flush=True)


def render_report(metrics: dict) -> str:
    lines = ["# Independent Email Benchmark", ""]
    classification = metrics.get("quality", {}).get("classification")
    if classification:
        lines.extend(
            [
                "## Classification Quality",
                f"- Samples: {classification['samples']}",
                f"- Accuracy: {classification['accuracy']:.4f}",
                f"- Macro F1: {classification['macro_f1']:.4f}",
                f"- Spam precision: {classification['spam_precision']:.4f}",
                f"- Spam recall: {classification['spam_recall']:.4f}",
                f"- Parse success: {classification['parse_success_rate']:.4f}",
                f"- Valid category: {classification['valid_category_rate']:.4f}",
                "",
            ]
        )
    by_task = metrics.get("speed", {}).get("by_task", {})
    if by_task:
        lines.extend(["## Per-task Speed", "", "| Task | p50 ms | p95 ms | p99 ms | req/s | output tok/s | parse success |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
        for task, values in by_task.items():
            latency = values["wall_latency_ms"]
            lines.append(
                f"| {task} | {latency['p50']:.2f} | {latency['p95']:.2f} | {latency['p99']:.2f} | "
                f"{values['requests_per_second']:.2f} | {values['output_tokens_per_second']:.2f} | "
                f"{values['parse_success_rate']:.4f} |"
            )
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
