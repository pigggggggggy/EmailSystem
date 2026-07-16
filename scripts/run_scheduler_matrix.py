#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVAL_SCRIPT = ROOT / "scripts" / "run_parallel_eval.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a max_num_batched_tokens x max_num_seqs vLLM evaluation matrix."
    )
    parser.add_argument("--run-root", default=None)
    parser.add_argument("--batched-tokens", type=int, nargs="+", default=[4096, 8192, 12288])
    parser.add_argument("--max-num-seqs", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument(
        "eval_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to run_parallel_eval.py; place them after --.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.eval_args:
        raise SystemExit("Pass run_parallel_eval.py arguments after --.")
    if any(value <= 0 for value in [*args.batched_tokens, *args.max_num_seqs]):
        raise SystemExit("Matrix values must be positive.")

    run_root = Path(args.run_root) if args.run_root else Path("outputs/runs") / datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S_scheduler_matrix"
    )
    run_root.mkdir(parents=True, exist_ok=True)
    forwarded = args.eval_args[1:] if args.eval_args[0] == "--" else args.eval_args
    results = []
    for token_budget in args.batched_tokens:
        for max_num_seqs in args.max_num_seqs:
            name = f"batched_{token_budget}_seqs_{max_num_seqs}"
            run_dir = run_root / name
            command = [
                sys.executable,
                str(EVAL_SCRIPT),
                *forwarded,
                "--max-num-batched-tokens",
                str(token_budget),
                "--max-num-seqs",
                str(max_num_seqs),
                "--run-dir",
                str(run_dir),
            ]
            print(f"\n=== {name} ===", flush=True)
            completed = subprocess.run(command, cwd=ROOT)
            results.append(
                {
                    "max_num_batched_tokens": token_budget,
                    "max_num_seqs": max_num_seqs,
                    "run_dir": str(run_dir),
                    "returncode": completed.returncode,
                    "metrics": read_metrics(run_dir),
                }
            )
            if completed.returncode:
                break
        if results and results[-1]["returncode"]:
            break

    (run_root / "matrix.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (run_root / "README.md").write_text(render_readme(results), encoding="utf-8")
    if results and results[-1]["returncode"]:
        raise SystemExit(results[-1]["returncode"])


def read_metrics(run_dir: Path) -> dict:
    path = run_dir / "metrics.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def render_readme(results: list[dict]) -> str:
    lines = [
        "# vLLM Scheduler Matrix",
        "",
        "| Max batched tokens | Max sequences | Accuracy | Status | Run |",
        "| ---: | ---: | ---: | --- | --- |",
    ]
    for item in results:
        status = "ok" if item["returncode"] == 0 else f"failed ({item['returncode']})"
        quality = item.get("metrics", {}).get("quality", {}).get("classification", {})
        accuracy = quality.get("accuracy")
        accuracy_text = f"{accuracy:.4f}" if isinstance(accuracy, (int, float)) else "-"
        lines.append(
            f"| {item['max_num_batched_tokens']} | {item['max_num_seqs']} | {accuracy_text} | "
            f"{status} | `{item['run_dir']}` |"
        )

    speed_rows = []
    for item in results:
        by_task = item.get("metrics", {}).get("speed", {}).get("by_task", {})
        for task, values in by_task.items():
            speed_rows.append((item, task, values))
    if speed_rows:
        lines.extend(
            [
                "",
                "## Per-task Throughput",
                "",
                "| Max batched tokens | Max sequences | Task | Req/s | Input tok/s | Output tok/s | Parse success |",
                "| ---: | ---: | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for item, task, values in speed_rows:
            lines.append(
                f"| {item['max_num_batched_tokens']} | {item['max_num_seqs']} | {task} | "
                f"{values.get('requests_per_second', 0.0):.2f} | "
                f"{values.get('input_tokens_per_second', 0.0):.2f} | "
                f"{values.get('output_tokens_per_second', 0.0):.2f} | "
                f"{values.get('parse_success_rate', 0.0):.4f} |"
            )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
