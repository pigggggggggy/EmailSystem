#!/usr/bin/env python3
"""Merge task-routed parallel benchmark runs into the standard report layout."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from email_system.io import read_jsonl, write_jsonl
from run_parallel_eval import render_parallel_report


TASK_RUNS = {
    "classify_email": "baseline_short",
    "summarize_email": "baseline_short",
    "extract_action_items": "ngram_action_items_st4",
    "draft_reply": "ngram_draft_reply_st6",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge task-routed n-gram outputs into the normal parallel-report format."
    )
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"Missing required routed output: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _merge_jsonl(paths: list[Path], destination: Path) -> None:
    rows = []
    for path in paths:
        if path.is_file():
            rows.extend(read_jsonl(path))
    if rows:
        write_jsonl(destination, rows)


def main() -> None:
    args = parse_args()
    root = Path(args.run_root)
    output_dir = Path(args.output_dir) if args.output_dir else root / "combined"
    output_dir.mkdir(parents=True, exist_ok=True)
    child_names = sorted(set(TASK_RUNS.values()))
    child_metrics = {name: _read_json(root / name / "metrics.json") for name in child_names}
    config = _read_json(root / "baseline_short" / "config.json")

    by_task = {}
    task_max_tokens = {}
    routing = {}
    for task, child in TASK_RUNS.items():
        speed = child_metrics[child].get("speed", {})
        values = speed.get("by_task", {}).get(task)
        if values is None:
            raise SystemExit(f"Missing speed result for {task} in {root / child}")
        by_task[task] = values
        task_max_tokens[task] = speed.get("task_max_tokens", {}).get(task)
        child_config = _read_json(root / child / "config.json")
        routing[task] = {
            "run_dir": str(root / child),
            "method": "baseline" if child == "baseline_short" else "ngram",
            "speculative_tokens": 0 if child == "baseline_short" else child_config["speculative_tokens"],
        }

    baseline_speed = child_metrics["baseline_short"].get("speed", {})
    metrics = {
        "quality": child_metrics["baseline_short"].get("quality", {}),
        "speed": {
            "samples_per_task": baseline_speed.get("samples_per_task"),
            "warmup_per_task": baseline_speed.get("warmup_per_task"),
            "batch_size": baseline_speed.get("batch_size"),
            "execution_mode": baseline_speed.get("execution_mode"),
            "queue_size": baseline_speed.get("queue_size"),
            "task_max_tokens": task_max_tokens,
            "by_task": by_task,
            "task_routing": routing,
        },
    }
    config.update(
        run_dir=str(output_dir), speed_tasks=list(TASK_RUNS), task_routing=routing,
        merged_from={name: str(root / name) for name in child_names},
    )
    (output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    route_lines = [
        "## Task Routing", "",
        "Each task is measured in a separate vLLM process so it can use its own speculative configuration. Compare the per-task rows directly with a normal parallel report; do not add queue wall times across rows.", "",
        "| Task | Method | Speculative tokens | Child run |",
        "| --- | --- | ---: | --- |",
    ]
    for task, values in routing.items():
        tokens = values["speculative_tokens"] or "--"
        route_lines.append(f"| {task} | {values['method']} | {tokens} | `{Path(values['run_dir']).name}` |")
    report = render_parallel_report(metrics, argparse.Namespace(**config))
    (output_dir / "report.md").write_text(report + "\n" + "\n".join(route_lines) + "\n", encoding="utf-8")

    _merge_jsonl([root / name / "speed_samples.jsonl" for name in child_names], output_dir / "speed_samples.jsonl")
    predictions = root / "baseline_short" / "classification_predictions.jsonl"
    if predictions.is_file():
        shutil.copy2(predictions, output_dir / "classification_predictions.jsonl")
    print(f"Wrote combined parallel-compatible report: {output_dir}")


if __name__ == "__main__":
    main()
