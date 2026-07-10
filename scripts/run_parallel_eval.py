#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from email_system.evaluation.independent_benchmark import (
    TASKS,
    _gold_category,
    _gold_spam_label,
    _latency_summary,
    _success_rate,
    _wrong_category,
    classification_quality_metrics,
    resolve_quality_mode,
    select_benchmark_rows,
    truncate_body_text,
)
from email_system.io import read_jsonl, write_jsonl
from email_system.models import build_llm_client
from email_system.models.chat_prompts import PROMPT_VERSION, messages_for_task
from email_system.models.vllm_client import truncate_token_ids
from email_system.schemas import Email
from email_system.skills.classify import LOW_CONFIDENCE_THRESHOLD, VALID_CATEGORIES
from email_system.skills.json_utils import ModelOutputParseError, parse_json_object


TASK_MAX_TOKENS = {
    "classify_email": 256,
    "summarize_email": 256,
    "extract_action_items": 256,
    "draft_reply": 512,
}
VALID_PRIORITIES = {"low", "normal", "high", "urgent"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch/parallel vLLM benchmark with optional EAGLE3.")
    parser.add_argument("--input", default="data/processed/spam_benchmark/test.jsonl")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--model-path", default="models/Qwen3-4B")
    parser.add_argument("--eagle3-model-path", default=None)
    parser.add_argument("--speculative-tokens", type=int, default=3)
    parser.add_argument("--quality-limit", type=int, default=None)
    parser.add_argument("--quality-mode", choices=["auto", "binary", "multiclass"], default="auto")
    parser.add_argument("--max-body-chars", type=int, default=6000)
    parser.add_argument("--speed-limit", type=int, default=100)
    parser.add_argument("--speed-warmup", type=int, default=2)
    parser.add_argument("--speed-tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--skip-quality", action="store_true")
    parser.add_argument("--skip-speed", action="store_true")
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--torch-dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument(
        "--use-compiled-graphs",
        action="store_true",
        help="Use vLLM compiled CUDA graphs instead of the memory-safe eager default.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if args.skip_quality and args.skip_speed:
        raise SystemExit("Both quality and speed phases are disabled.")

    rows = truncate_body_text(read_jsonl(args.input), args.max_body_chars)
    quality_mode = resolve_quality_mode(rows, args.quality_mode)
    quality_rows = select_benchmark_rows(rows, args.quality_limit, seed=args.seed, label_mode=quality_mode)
    speed_rows = select_benchmark_rows(rows, args.speed_limit, seed=args.seed)
    if not args.skip_quality and not quality_rows:
        raise SystemExit(f"No quality rows selected from {args.input}.")
    if not args.skip_speed and not speed_rows:
        raise SystemExit(f"No speed rows selected from {args.input}.")

    run_dir = Path(args.run_dir) if args.run_dir else Path("outputs/runs") / datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S_vllm_parallel"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["input_records"] = len(rows)
    config["resolved_quality_mode"] = quality_mode
    config["prompt_version"] = PROMPT_VERSION
    (run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Loading vLLM model={args.model_path}", flush=True)
    if args.eagle3_model_path:
        print(f"Using EAGLE3 draft={args.eagle3_model_path} spec_tokens={args.speculative_tokens}", flush=True)
    llm = build_llm_client(
        "vllm",
        model_path=args.model_path,
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
        predictions = run_batched_classification_quality(
            llm,
            quality_rows,
            quality_mode=quality_mode,
            batch_size=args.batch_size,
            show_progress=True,
        )
        write_jsonl(run_dir / "classification_predictions.jsonl", predictions)
        metrics["quality"] = {"classification": classification_quality_metrics(predictions)}

    if not args.skip_speed:
        samples, speed_metrics = run_batched_task_speed(
            llm,
            speed_rows,
            tasks=args.speed_tasks,
            batch_size=args.batch_size,
            warmup=args.speed_warmup,
            show_progress=True,
        )
        write_jsonl(run_dir / "speed_samples.jsonl", samples)
        metrics["speed"] = speed_metrics

    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (run_dir / "report.md").write_text(render_parallel_report(metrics, args), encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "metrics": metrics}, ensure_ascii=False, indent=2), flush=True)




def run_batched_classification_quality(
    llm,
    rows: list[dict],
    *,
    quality_mode: str,
    batch_size: int,
    show_progress: bool,
) -> list[dict]:
    predictions = []
    progress = _progress(len(rows), desc='quality', unit='email') if show_progress else None
    for batch in _chunks(rows, batch_size):
        emails = [Email.from_dict(row) for row in batch]
        results = generate_batch(llm, emails, task='classify_email', max_tokens=TASK_MAX_TOKENS['classify_email'])
        for row, email, result in zip(batch, emails, results):
            parsed = parse_classification(result['text'])
            predicted_category = str(parsed.get('category', 'automated_email'))
            invalid = bool(parsed.get('parse_error')) or predicted_category not in VALID_CATEGORIES
            confidence = float(parsed.get('confidence', 0.0))
            low_confidence = bool(parsed.get('low_confidence', confidence < LOW_CONFIDENCE_THRESHOLD))
            if quality_mode == 'multiclass':
                gold_label = _gold_category(row)
                predicted_label = predicted_category if not invalid else _wrong_category(gold_label)
            else:
                gold_label = _gold_spam_label(row)
                predicted_label = (('spam' if predicted_category == 'spam' else 'ham') if not invalid else ('ham' if gold_label == 'spam' else 'spam'))
            prediction = {
                'email_id': email.email_id,
                'quality_mode': quality_mode,
                'gold_label': gold_label,
                'predicted_label': predicted_label,
                'predicted_category': predicted_category,
                'confidence': confidence,
                'low_confidence': low_confidence,
                'accepted_prediction': not invalid and not low_confidence,
                'parse_error': parsed.get('parse_error'),
                'valid_category': not invalid,
                'usage': {
                    'text': result['text'],
                    'input_tokens': result['input_tokens'],
                    'output_tokens': result['output_tokens'],
                    'latency_ms': result['batch_wall_ms'],
                    'batch_size': result['batch_size'],
                    'batch_amortized_latency_ms': result['amortized_wall_ms'],
                },
            }
            if quality_mode == 'multiclass':
                prediction.update(gold_category=gold_label, scored_category=predicted_label)
            else:
                prediction.update(gold_spam_label=gold_label, predicted_spam_label=predicted_label)
            predictions.append(prediction)
        if progress is not None:
            progress.update(len(batch))
    if progress is not None:
        progress.close()
    return predictions


def run_batched_task_speed(
    llm,
    rows: list[dict],
    *,
    tasks: Iterable[str],
    batch_size: int,
    warmup: int,
    show_progress: bool,
) -> tuple[list[dict], dict]:
    task_names = list(tasks)
    unknown = sorted(set(task_names) - set(TASKS))
    if unknown:
        raise ValueError(f"unsupported speed tasks: {', '.join(unknown)}")

    progress = _progress(len(rows) * len(task_names), desc='parallel-speed', unit='request') if show_progress else None
    samples = []
    batch_samples = []
    for task in task_names:
        max_tokens = TASK_MAX_TOKENS[task]
        warmup_emails = [Email.from_dict(row) for row in rows[:warmup]]
        if warmup_emails:
            generate_batch(llm, warmup_emails, task=task, max_tokens=max_tokens)
        for batch_index, batch in enumerate(_chunks(rows, batch_size), start=1):
            emails = [Email.from_dict(row) for row in batch]
            results = generate_batch(llm, emails, task=task, max_tokens=max_tokens)
            batch_wall_ms = results[0]['batch_wall_ms'] if results else 0.0
            batch_samples.append({
                'task': task,
                'batch_index': batch_index,
                'batch_size': len(results),
                'wall_latency_ms': batch_wall_ms,
                'input_tokens': sum(item['input_tokens'] for item in results),
                'output_tokens': sum(item['output_tokens'] for item in results),
            })
            for email, result in zip(emails, results):
                samples.append({
                    'task': task,
                    'email_id': email.email_id,
                    'body_chars': len(email.body_text),
                    'batch_size': result['batch_size'],
                    'batch_wall_latency_ms': result['batch_wall_ms'],
                    'amortized_wall_latency_ms': result['amortized_wall_ms'],
                    'input_tokens': result['input_tokens'],
                    'output_tokens': result['output_tokens'],
                    'parse_error': parse_error_for_task(task, result['text']),
                })
            if progress is not None:
                progress.update(len(batch))
    if progress is not None:
        progress.close()

    by_task = {}
    for task in task_names:
        task_samples = [sample for sample in samples if sample['task'] == task]
        task_batches = [sample for sample in batch_samples if sample['task'] == task]
        by_task[task] = parallel_speed_metrics(task_samples, task_batches)
    return samples, {'samples_per_task': len(rows), 'warmup_per_task': warmup, 'batch_size': batch_size, 'by_task': by_task}


def generate_batch(llm, emails: list[Email], *, task: str, max_tokens: int) -> list[dict]:
    prompts = []
    input_token_counts = []
    for email in emails:
        prompt = email.to_prompt_text()
        if task == "draft_reply":
            prompt = prompt + "\n\nContext: {}"
        rendered = llm._render_prompt(messages_for_task(prompt, task))
        token_ids = llm.tokenizer.encode(rendered, add_special_tokens=False)
        if llm.max_model_len is not None:
            token_ids = truncate_token_ids(token_ids, llm.max_model_len - max_tokens)
        prompts.append({"prompt_token_ids": token_ids})
        input_token_counts.append(len(token_ids))

    sampling = llm.sampling_params_cls(temperature=0.0, max_tokens=max_tokens)
    start = time.perf_counter()
    outputs = llm.llm.generate(prompts, sampling, use_tqdm=False)
    batch_wall_ms = (time.perf_counter() - start) * 1000
    batch_size = len(outputs)
    amortized = batch_wall_ms / batch_size if batch_size else 0.0
    results = []
    for output, input_tokens in zip(outputs, input_token_counts):
        completion = output.outputs[0]
        results.append(
            {
                "text": completion.text.strip(),
                "input_tokens": input_tokens,
                "output_tokens": len(getattr(completion, "token_ids", []) or []),
                "finish_reason": getattr(completion, "finish_reason", None),
                "batch_size": batch_size,
                "batch_wall_ms": batch_wall_ms,
                "amortized_wall_ms": amortized,
            }
        )
    return results


def parse_classification(text: str) -> dict:
    try:
        data = parse_json_object(text)
    except ModelOutputParseError as exc:
        return {
            "category": "automated_email",
            "priority": "normal",
            "confidence": 0.0,
            "parse_error": str(exc),
            "raw_model_output": text,
            "low_confidence": True,
        }
    category = data.get("category")
    priority = data.get("priority")
    confidence = data.get("confidence")
    errors = []
    if category not in VALID_CATEGORIES:
        errors.append(f"invalid category: {category!r}")
    if priority not in VALID_PRIORITIES:
        errors.append(f"invalid priority: {priority!r}")
    valid_confidence = not isinstance(confidence, bool) and isinstance(confidence, (int, float)) and 0 <= confidence <= 1
    if not valid_confidence:
        errors.append(f"invalid confidence: {confidence!r}")
    parsed = {
        "category": category if category in VALID_CATEGORIES else "automated_email",
        "priority": priority if priority in VALID_PRIORITIES else "normal",
        "confidence": float(confidence) if valid_confidence else 0.0,
    }
    if errors:
        parsed["parse_error"] = "; ".join(errors)
        parsed["raw_model_output"] = text
    parsed["low_confidence"] = parsed["confidence"] < LOW_CONFIDENCE_THRESHOLD
    return parsed


def parse_error_for_task(task: str, text: str) -> str | None:
    if task == "draft_reply":
        return None if text.strip() else "empty reply"
    try:
        data = parse_json_object(text)
    except ModelOutputParseError as exc:
        return str(exc)
    if task == "classify_email":
        return parse_classification(text).get("parse_error")
    if task == "summarize_email" and not str(data.get("summary", "")).strip():
        return "missing summary"
    if task == "extract_action_items" and not isinstance(data.get("action_items"), list):
        return "missing action_items list"
    return None


def parallel_speed_metrics(samples: list[dict], batches: list[dict]) -> dict:
    total_wall_seconds = sum(float(row["wall_latency_ms"]) for row in batches) / 1000
    input_tokens = sum(int(row["input_tokens"]) for row in samples)
    output_tokens = sum(int(row["output_tokens"]) for row in samples)
    amortized_latencies = [float(row["amortized_wall_latency_ms"]) for row in samples]
    batch_latencies = [float(row["wall_latency_ms"]) for row in batches]
    return {
        "requests": len(samples),
        "batches": len(batches),
        "batch_size_effective_mean": (sum(row["batch_size"] for row in batches) / len(batches)) if batches else 0.0,
        "batch_wall_latency_ms": _latency_summary(batch_latencies),
        "amortized_request_latency_ms": _latency_summary(amortized_latencies),
        "requests_per_second": len(samples) / total_wall_seconds if total_wall_seconds else 0.0,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_tokens_per_second": input_tokens / total_wall_seconds if total_wall_seconds else 0.0,
        "output_tokens_per_second": output_tokens / total_wall_seconds if total_wall_seconds else 0.0,
        "parse_success_rate": _success_rate(row.get("parse_error") for row in samples),
    }


def render_parallel_report(metrics: dict, args: argparse.Namespace) -> str:
    return json.dumps(metrics, ensure_ascii=False, indent=2)


def _chunks(rows: list[dict], size: int) -> Iterable[list[dict]]:
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def _progress(total: int, *, desc: str, unit: str):
    try:
        from tqdm.auto import tqdm
    except ImportError as exc:
        raise RuntimeError("Progress display requires tqdm. Install it with: pip install tqdm") from exc
    return tqdm(total=total, desc=desc, unit=unit, dynamic_ncols=True)


if __name__ == "__main__":
    main()
