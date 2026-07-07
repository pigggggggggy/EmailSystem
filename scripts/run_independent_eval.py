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
    classification_quality_metrics,
    run_classification_quality,
    resolve_quality_mode,
    run_task_speed,
    select_benchmark_rows,
    truncate_body_text,
)
from email_system.io import read_jsonl, write_jsonl
from email_system.models import build_llm_client
from email_system.models.chat_prompts import PROMPT_VERSION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Independent classification-quality and per-task speed benchmark.")
    parser.add_argument("--input", default="data/processed/spam_benchmark/test.jsonl")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--backend", default="vllm", choices=["mock", "transformers", "vllm"])
    parser.add_argument("--model-path", default="models/Qwen3-4B")
    parser.add_argument("--eagle3-model-path", default=None)
    parser.add_argument("--speculative-tokens", type=int, default=3)
    parser.add_argument("--quality-limit", type=int, default=None)
    parser.add_argument("--quality-mode", choices=["auto", "binary", "multiclass"], default="auto")
    parser.add_argument("--max-body-chars", type=int, default=6000)
    parser.add_argument("--speed-limit", type=int, default=100)
    parser.add_argument("--speed-warmup", type=int, default=2)
    parser.add_argument("--speed-tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--skip-quality", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume classification from an existing --run-dir")
    parser.add_argument(
        "--use-compiled-graphs",
        action="store_true",
        help="Use vLLM compiled CUDA graphs instead of the memory-safe eager default",
    )
    parser.add_argument("--skip-speed", action="store_true")
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.skip_quality and args.skip_speed:
        raise SystemExit("Both quality and speed phases are disabled.")
    rows = truncate_body_text(read_jsonl(args.input), args.max_body_chars)
    resolved_quality_mode = resolve_quality_mode(rows, args.quality_mode)
    quality_rows = select_benchmark_rows(
        rows, args.quality_limit, seed=args.seed, label_mode=resolved_quality_mode
    )
    speed_rows = select_benchmark_rows(rows, args.speed_limit, seed=args.seed)
    if not args.skip_quality and not quality_rows:
        raise SystemExit(
            f"No quality rows selected from {args.input}. "
            "For multiclass evaluation, check that the file is non-empty and each row has labels.category."
        )
    if not args.skip_speed and not speed_rows:
        raise SystemExit(f"No speed rows selected from {args.input}. Check that the input file is non-empty.")
    run_dir = Path(args.run_dir) if args.run_dir else Path("outputs/runs") / datetime.now(timezone.utc).strftime(
        f"%Y%m%d_%H%M%S_{args.backend}_independent"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["resume"] = False
    config["input_records"] = len(rows)
    config["resolved_quality_mode"] = resolved_quality_mode
    config["prompt_version"] = PROMPT_VERSION
    _prepare_config(run_dir / "config.json", config, resume=args.resume)

    print(f"Loading backend={args.backend} model={args.model_path}", flush=True)
    llm = build_llm_client(
        args.backend,
        model_path=args.model_path,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=not args.use_compiled_graphs,
        speculative_model_path=args.eagle3_model_path,
        speculative_tokens=args.speculative_tokens,
    )

    metrics = {}
    if not args.skip_quality:
        print(f"Running classification quality on {len(quality_rows)} emails...", flush=True)
        predictions, quality_metrics = _run_quality_with_checkpoints(
            llm,
            quality_rows,
            run_dir / "classification_predictions.jsonl",
            resume=args.resume,
            quality_mode=resolved_quality_mode,
        )
        metrics["quality"] = {"classification": quality_metrics}

    if not args.skip_speed:
        print(f"Running per-task speed on {len(speed_rows)} emails x {len(args.speed_tasks)} tasks...", flush=True)
        samples, speed_metrics = run_task_speed(
            llm,
            speed_rows,
            tasks=args.speed_tasks,
            warmup=args.speed_warmup,
            show_progress=True,
        )
        write_jsonl(run_dir / "speed_samples.jsonl", samples)
        metrics["speed"] = speed_metrics

    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (run_dir / "report.md").write_text(render_report(metrics), encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "metrics": metrics}, ensure_ascii=False, indent=2))


def _prepare_config(path: Path, config: dict, *, resume: bool) -> None:
    if resume:
        if not path.exists():
            raise SystemExit("Cannot resume: config.json is missing from --run-dir.")
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != config:
            raise SystemExit("Cannot resume: benchmark configuration or prompt version has changed.")
        return
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _run_quality_with_checkpoints(
    llm, rows: list[dict], output_path: Path, *, resume: bool, quality_mode: str
):
    existing = read_jsonl(output_path) if resume and output_path.exists() else []
    target_ids = [str(row["email_id"]) for row in rows]
    target_id_set = set(target_ids)
    existing_by_id = {str(row["email_id"]): row for row in existing}
    unknown_ids = set(existing_by_id) - target_id_set
    if unknown_ids:
        raise SystemExit("Resume file contains predictions outside the selected quality sample.")
    if len(existing_by_id) != len(existing):
        raise SystemExit("Resume file contains duplicate email IDs.")

    completed = [existing_by_id[email_id] for email_id in target_ids if email_id in existing_by_id]
    remaining = [row for row in rows if str(row["email_id"]) not in existing_by_id]
    if not resume:
        output_path.write_text("", encoding="utf-8")
    elif completed:
        print(f"Resuming classification from {len(completed)} completed emails.", flush=True)

    checkpoint_size = 100
    with output_path.open("a", encoding="utf-8") as handle:
        for start in range(0, len(remaining), checkpoint_size):
            batch = remaining[start : start + checkpoint_size]
            predictions, _ = run_classification_quality(llm, batch, quality_mode=quality_mode)
            for prediction in predictions:
                handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            handle.flush()
            completed.extend(predictions)
            print(f"quality: {len(completed)}/{len(rows)}", flush=True)

    ordered = {str(row["email_id"]): row for row in completed}
    predictions = [ordered[email_id] for email_id in target_ids]
    return predictions, classification_quality_metrics(predictions)


def render_report(metrics: dict) -> str:
    lines = ["# Independent Email Benchmark", ""]
    classification = metrics.get("quality", {}).get("classification")
    if classification:
        lines.extend(
            [
                "## Classification Quality",
                f"- Samples: {classification['samples']}",
                f"- Accuracy: {classification['accuracy']:.4f}",
                f"- Mode: {classification.get('quality_mode', 'binary')}",
                f"- Macro F1: {classification['macro_f1']:.4f}",
                f"- Parse success: {classification['parse_success_rate']:.4f}",
                f"- Valid category: {classification['valid_category_rate']:.4f}",
                f"- Low-confidence rate (<{classification['confidence_threshold']:.2f}): {classification['low_confidence_rate']:.4f}",
                f"- Auto-accepted coverage: {classification['accepted_coverage']:.4f}",
                f"- Auto-accepted accuracy: {classification['accepted_accuracy']:.4f}",
                "",
            ]
        )
        if classification.get("quality_mode", "binary") == "binary":
            lines[-1:-1] = [
                f"- Spam precision: {classification['spam_precision']:.4f}",
                f"- Spam recall: {classification['spam_recall']:.4f}",
            ]
        else:
            lines.extend(["### Per-class Quality", "", "| Category | Precision | Recall | F1 | Support |", "| --- | ---: | ---: | ---: | ---: |"])
            for category, values in classification.get("per_class", {}).items():
                lines.append(
                    f"| {category} | {values['precision']:.4f} | {values['recall']:.4f} | "
                    f"{values['f1']:.4f} | {values['support']} |"
                )
            lines.append("")
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
