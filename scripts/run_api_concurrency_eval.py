#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from email_system.evaluation.independent_benchmark import select_benchmark_rows, truncate_body_text
from email_system.io import read_jsonl, write_jsonl
from email_system.models.chat_prompts import PROMPT_VERSION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HTTP concurrency benchmark for the EmailSystem FastAPI app.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--endpoint", default="/api/process")
    parser.add_argument("--input", default="data/processed/spam_benchmark/test.jsonl")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--max-body-chars", type=int, default=6000)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--seed", type=int, default=20260710)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = truncate_body_text(read_jsonl(args.input), args.max_body_chars)
    rows = select_benchmark_rows(rows, args.limit, seed=args.seed, label_mode="binary")
    if not rows:
        raise SystemExit(f"No rows selected from {args.input}.")
    if any(value <= 0 for value in args.concurrency):
        raise SystemExit("--concurrency values must be positive")

    run_dir = Path(args.run_dir) if args.run_dir else Path("outputs/runs") / datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S_api_concurrency"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["input_records"] = len(read_jsonl(args.input))
    config["selected_records"] = len(rows)
    config["prompt_version"] = PROMPT_VERSION
    (run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    results = asyncio.run(run_all(args, rows))
    all_samples = [sample for result in results for sample in result["samples"]]
    metrics = {
        "endpoint": args.base_url.rstrip("/") + args.endpoint,
        "selected_records": len(rows),
        "by_concurrency": {str(result["concurrency"]): result["metrics"] for result in results},
    }
    write_jsonl(run_dir / "samples.jsonl", all_samples)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (run_dir / "report.md").write_text(render_report(metrics), encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "metrics": metrics}, ensure_ascii=False, indent=2), flush=True)


async def run_all(args: argparse.Namespace, rows: list[dict]) -> list[dict]:
    try:
        import httpx
    except ImportError as exc:
        raise SystemExit("Missing dependency: install httpx to run HTTP concurrency benchmarks.") from exc

    endpoint = args.base_url.rstrip("/") + args.endpoint
    timeout = httpx.Timeout(args.timeout)
    limits = httpx.Limits(max_connections=max(args.concurrency), max_keepalive_connections=max(args.concurrency))
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        health_url = args.base_url.rstrip("/") + "/api/health"
        try:
            health = await client.get(health_url)
            health.raise_for_status()
            print(f"health: {health.json()}", flush=True)
        except Exception as exc:
            raise SystemExit(f"API health check failed at {health_url}: {type(exc).__name__}: {exc}") from exc

        results = []
        for concurrency in args.concurrency:
            if args.warmup:
                warmup_rows = rows[: min(args.warmup, len(rows))]
                await run_once(client, endpoint, warmup_rows, concurrency=concurrency, phase="warmup")
            print(f"Running concurrency={concurrency} requests={len(rows)}", flush=True)
            samples, wall_seconds = await run_once(client, endpoint, rows, concurrency=concurrency, phase="measure")
            metrics = summarize_samples(samples, wall_seconds)
            results.append({"concurrency": concurrency, "metrics": metrics, "samples": samples})
    return results


async def run_once(client, endpoint: str, rows: list[dict], *, concurrency: int, phase: str) -> tuple[list[dict], float]:
    semaphore = asyncio.Semaphore(concurrency)
    start = time.perf_counter()
    tasks = [
        asyncio.create_task(post_email(client, endpoint, row, semaphore=semaphore, concurrency=concurrency, phase=phase))
        for row in rows
    ]
    samples = await asyncio.gather(*tasks)
    return samples, time.perf_counter() - start


async def post_email(client, endpoint: str, row: dict, *, semaphore: asyncio.Semaphore, concurrency: int, phase: str) -> dict:
    payload = email_payload(row)
    start = time.perf_counter()
    status_code = None
    error = None
    response_json: dict[str, Any] | None = None
    async with semaphore:
        try:
            response = await client.post(endpoint, json=payload)
            status_code = response.status_code
            response.raise_for_status()
            response_json = response.json()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    latency_ms = (time.perf_counter() - start) * 1000
    output = (response_json or {}).get("output", {})
    return {
        "phase": phase,
        "concurrency": concurrency,
        "email_id": str(row.get("email_id", "")),
        "status_code": status_code,
        "ok": error is None,
        "error": error,
        "latency_ms": latency_ms,
        "api_elapsed_ms": (response_json or {}).get("elapsed_ms"),
        "category": output.get("category"),
        "route": output.get("route"),
        "delivery_status": output.get("delivery_status"),
    }


def email_payload(row: dict) -> dict:
    to_values = row.get("to") or []
    if isinstance(to_values, str):
        to_values = [to_values]
    cc_values = row.get("cc") or []
    if isinstance(cc_values, str):
        cc_values = [cc_values]
    return {
        "email_id": str(row.get("email_id", "")),
        "thread_id": row.get("thread_id"),
        "subject": str(row.get("subject", "")),
        "sender": str(row.get("from") or row.get("sender", "")),
        "to": [str(value) for value in to_values],
        "cc": [str(value) for value in cc_values],
        "timestamp": row.get("timestamp"),
        "body_text": str(row.get("body_text", "")),
    }


def summarize_samples(samples: list[dict], wall_seconds: float) -> dict:
    measured = [sample for sample in samples if sample["phase"] == "measure"]
    latencies = [float(sample["latency_ms"]) for sample in measured]
    api_latencies = [float(sample["api_elapsed_ms"]) for sample in measured if sample.get("api_elapsed_ms") is not None]
    ok_count = sum(sample["ok"] for sample in measured)
    total = len(measured)
    return {
        "requests": total,
        "ok": ok_count,
        "errors": total - ok_count,
        "error_rate": (total - ok_count) / total if total else 0.0,
        "wall_seconds": wall_seconds,
        "requests_per_second": total / wall_seconds if wall_seconds else 0.0,
        "latency_ms": latency_summary(latencies),
        "api_elapsed_ms": latency_summary(api_latencies),
    }


def latency_summary(values: list[float]) -> dict:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "mean": statistics.mean(values),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": max(values),
    }


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * pct / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def render_report(metrics: dict) -> str:
    lines = [
        "# API Concurrency Benchmark",
        "",
        f"- Endpoint: `{metrics['endpoint']}`",
        f"- Requests per concurrency: {metrics['selected_records']}",
        "",
        "| Concurrency | req/s | p50 ms | p95 ms | p99 ms | errors | error rate |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for concurrency, values in metrics["by_concurrency"].items():
        latency = values["latency_ms"]
        lines.append(
            f"| {concurrency} | {values['requests_per_second']:.2f} | {latency['p50']:.2f} | "
            f"{latency['p95']:.2f} | {latency['p99']:.2f} | {values['errors']} | {values['error_rate']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
