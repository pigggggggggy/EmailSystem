#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "scripts" / "run_parallel_eval.py"
TASKS = ("classify_email", "summarize_email", "extract_action_items", "draft_reply")
SPEC_OPTIONS = {"--eagle3-model-path", "--ngram-prompt-lookup-min", "--ngram-prompt-lookup-max", "--speculative-tokens"}

def parse_args():
    p = argparse.ArgumentParser(description="Benchmark speculative-token counts independently per task.")
    p.add_argument("--method", choices=("ngram", "eagle3"), required=True)
    p.add_argument("--run-root", default=None)
    p.add_argument("--speculative-tokens", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--tasks", choices=TASKS, nargs="+", default=list(TASKS))
    p.add_argument("--no-baseline", action="store_true")
    p.add_argument("--best-by", choices=("requests_per_second", "output_tokens_per_second", "amortized_request_latency_ms"), default="requests_per_second")
    p.add_argument("--stop-on-error", action="store_true")
    p.add_argument("--render-only", action="store_true", help="Rebuild report.md from an existing matrix.json.")
    p.add_argument("eval_args", nargs=argparse.REMAINDER)
    return p.parse_args()

def main():
    args = parse_args()
    forwarded = args.eval_args[1:] if args.eval_args[:1] == ["--"] else args.eval_args
    if args.render_only:
        if not args.run_root:
            raise SystemExit("--render-only requires --run-root.")
        root = Path(args.run_root)
        matrix_path = root / "matrix.json"
        if not matrix_path.exists():
            raise SystemExit(f"Missing {matrix_path}.")
        write_results(root, json.loads(matrix_path.read_text(encoding="utf-8")), args)
        print(f"Rebuilt {root / 'report.md'}", flush=True)
        return

    validate(args, forwarded)
    root = Path(args.run_root) if args.run_root else Path("outputs/runs") / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_speculative_tokens_matrix")
    root.mkdir(parents=True, exist_ok=True)
    variants = ([] if args.no_baseline else [None]) + args.speculative_tokens
    results = []
    for task in args.tasks:
        for tokens in variants:
            name = "baseline" if tokens is None else f"spec_tokens_{tokens}"
            run_dir = root / task / name
            cmd = command(forwarded, task, tokens, run_dir)
            print(f"\n=== {task}: {name} ===\n{shlex.join(cmd)}", flush=True)
            done = subprocess.run(cmd, cwd=ROOT)
            results.append({"task": task, "method": "baseline" if tokens is None else args.method, "speculative_tokens": tokens, "run_dir": str(run_dir), "returncode": done.returncode, "metrics": read_metrics(run_dir)})
            if done.returncode and args.stop_on_error:
                write_results(root, results, args)
                raise SystemExit(done.returncode)
    write_results(root, results, args)
    if any(row["returncode"] for row in results):
        raise SystemExit("Some matrix runs failed; inspect matrix.json.")

def validate(args, forwarded):
    if not forwarded:
        raise SystemExit("Pass common run_parallel_eval.py arguments after --.")
    if "--skip-speed" in forwarded or any(x <= 0 for x in args.speculative_tokens):
        raise SystemExit("Do not pass --skip-speed; speculative-token values must be positive.")
    if len(set(args.speculative_tokens)) != len(args.speculative_tokens):
        raise SystemExit("Speculative-token values must be unique.")
    eagle = has(forwarded, "--eagle3-model-path")
    ngram = has(forwarded, "--ngram-prompt-lookup-min") or has(forwarded, "--ngram-prompt-lookup-max")
    if args.method == "ngram" and (not ngram or eagle):
        raise SystemExit("n-gram requires lookup arguments and cannot include an EAGLE3 model.")
    if args.method == "eagle3" and not eagle:
        raise SystemExit("eagle3 requires --eagle3-model-path.")

def has(values, option):
    return any(value == option or value.startswith(f"{option}=") for value in values)

def without_speculation(values):
    result, index = [], 0
    while index < len(values):
        value = values[index]
        if value.split("=", 1)[0] in SPEC_OPTIONS:
            index += 1 if "=" in value else 2
        else:
            result.append(value)
            index += 1
    return result

def command(forwarded, task, tokens, run_dir):
    common = forwarded if tokens is not None else without_speculation(forwarded)
    result = [sys.executable, str(EVAL), *common, "--skip-quality", "--speed-tasks", task, "--run-dir", str(run_dir)]
    if tokens is not None:
        result += ["--speculative-tokens", str(tokens)]
    return result

def read_metrics(run_dir):
    path = run_dir / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

def speed(row):
    return row.get("metrics", {}).get("speed", {}).get("by_task", {}).get(row["task"], {}) if row else {}

def best(rows, task, metric):
    candidates = [row for row in rows if row["task"] == task and not row["returncode"] and row["speculative_tokens"] is not None]
    if not candidates:
        return None
    return sorted(candidates, key=lambda row: speed(row).get(metric, 0.0), reverse=metric != "amortized_request_latency_ms")[0]

def number(value, precision=2):
    return f"{value:.{precision}f}" if isinstance(value, (int, float)) else "--"

def hit(values):
    rate = values.get("speculative_decoding", {}).get("draft_acceptance_rate")
    return f"{rate:.2%}" if isinstance(rate, (int, float)) else "--"

def write_results(root, rows, args):
    (root / "matrix.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Speculative-token Matrix", "", f"- Method: {args.method}", f"- Selection metric: {args.best_by}", "- Each task uses an independent vLLM process.", "- Hit rate = accepted draft tokens / drafted tokens.", "", "## Best Per Task", "", "| Task | Baseline Req/s | Best spec tokens | Best Req/s | Change | Hit rate | Run |", "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for task in args.tasks:
        base = next((row for row in rows if row["task"] == task and row["speculative_tokens"] is None), None)
        winner = best(rows, task, args.best_by)
        base_rps, values = speed(base).get("requests_per_second"), speed(winner)
        rps = values.get("requests_per_second")
        change = rps / base_rps - 1 if isinstance(rps, (int, float)) and base_rps else None
        lines.append(f"| {task} | {number(base_rps)} | {winner['speculative_tokens'] if winner else '--'} | {number(rps)} | {f'{change:+.2%}' if change is not None else '--'} | {hit(values)} | {winner['run_dir'] if winner else '--'} |")
    lines += ["", "## Full Results", "", "| Task | Method | Spec tokens | Req/s | Amortized ms/request | Output tok/s | Draft / accepted | Hit rate | Parse success | Status | Run |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |"]
    for row in rows:
        values, spec = speed(row), speed(row).get("speculative_decoding", {})
        drafted, accepted = spec.get("drafted_tokens"), spec.get("accepted_tokens")
        draft = f"{drafted} / {accepted}" if drafted is not None else "--"
        status = "ok" if not row["returncode"] else "failed ({})".format(row["returncode"])
        lines.append("| {task} | {method} | {tokens} | {rps} | {latency} | {out_tps} | {draft} | {hit_rate} | {parse} | {status} | {run_dir} |".format(task=row["task"], method=row["method"], tokens=row["speculative_tokens"] if row["speculative_tokens"] is not None else "--", rps=number(values.get("requests_per_second")), latency=number(values.get("amortized_request_latency_ms", {}).get("p50")), out_tps=number(values.get("output_tokens_per_second")), draft=draft, hit_rate=hit(values), parse=number(values.get("parse_success_rate"), 4), status=status, run_dir=row["run_dir"]))
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
