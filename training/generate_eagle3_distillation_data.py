#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from email_system.models.chat_prompts import PROMPT_VERSION, messages_for_task
from email_system.models.vllm_client import VLLMClient, truncate_token_ids
from email_system.skills.classify import VALID_CATEGORIES
from email_system.skills.json_utils import parse_json_object

TASKS = ("classify_email", "summarize_email", "extract_action_items", "draft_reply")
TASK_MAX_TOKENS = {
    "classify_email": 128,
    "summarize_email": 192,
    "extract_action_items": 256,
    "draft_reply": 384,
}
DEFAULT_INPUT_DIRS = ("data/processed/spam_benchmark", "data/processed/phishing_benchmark")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distill mixed email tasks from a target model for EAGLE3.")
    parser.add_argument("--input-dir", action="append", default=None)
    parser.add_argument("--model-path", default="models/Qwen3-4B-email-classifier-ckpt1563")
    parser.add_argument("--output-dir", default="data/finetune/eagle3_mixed")
    parser.add_argument("--train-per-task", type=int, default=2500)
    parser.add_argument("--validation-per-task", type=int, default=250)
    parser.add_argument("--max-body-chars", type=int, default=800)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--training-max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--oversample-factor", type=float, default=2.0)
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=20260702)
    parser.add_argument("--task", action="append", choices=TASKS, default=None, help="Only distill selected task(s). Repeat to include multiple tasks.")
    parser.add_argument("--retry-rejected-tasks", action="append", choices=TASKS, default=None, help="Ignore previous rejected rows for selected task(s), so they can be regenerated.")
    parser.add_argument("--allow-config-change", action="store_true", help="Allow updating generation_config.json when refilling selected tasks in an existing output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_args(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_dirs = [Path(value) for value in (args.input_dir or DEFAULT_INPUT_DIRS)]
    active_tasks = tuple(args.task or TASKS)
    retry_rejected_tasks = set(args.retry_rejected_tasks or [])
    generation_config = {
        "prompt_version": PROMPT_VERSION,
        "teacher_model": args.model_path,
        "source_dirs": [str(path) for path in input_dirs],
        "train_per_task": args.train_per_task,
        "validation_per_task": args.validation_per_task,
        "max_body_chars": args.max_body_chars,
        "max_model_len": args.max_model_len,
        "training_max_length": args.training_max_length,
        "oversample_factor": args.oversample_factor,
        "seed": args.seed,
    }
    prepare_generation_config(output_dir / "generation_config.json", generation_config, allow_change=args.allow_config_change)

    split_candidates = {}
    for split, limit in (("train", args.train_per_task), ("validation", args.validation_per_task)):
        rows = collect_rows(input_dirs, split)
        candidate_limit = math.ceil(limit * args.oversample_factor)
        selected = stable_sample(rows, limit=candidate_limit, seed=args.seed + (split == "validation"))
        split_candidates[split] = build_candidates(selected, max_body_chars=args.max_body_chars, tasks=active_tasks)

    client = VLLMClient(
        args.model_path,
        dtype=args.torch_dtype,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
    )
    counts = {}
    for split in ("train", "validation"):
        counts[split] = generate_split(
            client,
            split_candidates[split],
            output_dir / f"{split}.jsonl",
            output_dir / f"{split}_rejected.jsonl",
            batch_size=args.batch_size,
            target_per_task=args.train_per_task if split == "train" else args.validation_per_task,
            training_max_length=args.training_max_length,
            tasks=active_tasks,
            retry_rejected_tasks=retry_rejected_tasks,
        )

    manifest = {
        "task": "eagle3_mixed_parent_distillation",
        "format": "angelslim_conversations_v1",
        "prompt_version": PROMPT_VERSION,
        "teacher_model": args.model_path,
        "source_dirs": [str(path) for path in input_dirs],
        "tasks": list(active_tasks),
        "task_max_tokens": TASK_MAX_TOKENS,
        "train_per_task": args.train_per_task,
        "validation_per_task": args.validation_per_task,
        "max_body_chars": args.max_body_chars,
        "max_model_len": args.max_model_len,
        "training_max_length": args.training_max_length,
        "oversample_factor": args.oversample_factor,
        "seed": args.seed,
        "counts": counts,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


def prepare_generation_config(path: Path, config: dict, *, allow_change: bool = False) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != config:
            if not allow_change:
                raise SystemExit(
                    f"Existing distillation configuration differs: {path}. "
                    "Use another --output-dir, restore the original arguments, or pass --allow-config-change "
                    "when intentionally refilling selected tasks."
                )
            updated = dict(config)
            updated["previous_generation_config"] = existing
            path.write_text(json.dumps(updated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def collect_rows(input_dirs: list[Path], split: str) -> list[dict]:
    rows_by_id = {}
    for input_dir in input_dirs:
        path = input_dir / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"missing split file: {path}")
        for index, row in enumerate(read_jsonl(path)):
            row_id = str(row.get("email_id") or f"{path}:{index}")
            rows_by_id.setdefault(row_id, row)
    return list(rows_by_id.values())


def stable_sample(rows: Iterable[dict], *, limit: int, seed: int) -> list[dict]:
    ordered = sorted(rows, key=lambda row: stable_key(seed, str(row.get("email_id", ""))))
    return ordered[:limit] if limit > 0 else ordered


def build_candidates(rows: Iterable[dict], *, max_body_chars: int, tasks: Iterable[str] = TASKS) -> list[dict]:
    candidates = []
    for row in rows:
        email_id = str(row.get("email_id", ""))
        prompt = email_prompt(row, max_body_chars=max_body_chars)
        for task in tasks:
            candidates.append(
                {
                    "id": f"{email_id}:{task}",
                    "email_id": email_id,
                    "task": task,
                    "messages": messages_for_task(prompt, task),
                }
            )
    return candidates


def generate_split(
    client: VLLMClient,
    candidates: list[dict],
    output_path: Path,
    rejected_path: Path,
    *,
    batch_size: int,
    target_per_task: int,
    training_max_length: int,
    tasks: Iterable[str] = TASKS,
    retry_rejected_tasks: set[str] | None = None,
) -> dict:
    task_names = tuple(tasks)
    retry_rejected_tasks = retry_rejected_tasks or set()
    completed = load_completed_ids(output_path) | load_completed_ids(rejected_path, exclude_tasks=retry_rejected_tasks)
    pending = [item for item in candidates if item["id"] not in completed]
    accepted_rows = list(read_jsonl(output_path)) if output_path.exists() else []
    accepted_by_task = {task: sum(row.get("task") == task for row in accepted_rows) for task in task_names}
    accepted_count = len(accepted_rows)
    rejected_count = count_jsonl(rejected_path)
    print(
        f"Generating {output_path.stem}: pending={len(pending)} "
        f"accepted={accepted_count} rejected={rejected_count}",
        flush=True,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as accepted, rejected_path.open("a", encoding="utf-8") as rejected:
        for task in tasks:
            task_pending = [item for item in pending if item["task"] == task]
            max_tokens = TASK_MAX_TOKENS[task]
            if accepted_by_task[task] >= target_per_task:
                continue
            for start in range(0, len(task_pending), batch_size):
                batch = task_pending[start : start + batch_size]
                texts, finish_reasons = generate_batch(client, batch, max_tokens=max_tokens)
                for item, text, finish_reason in zip(batch, texts, finish_reasons):
                    if accepted_by_task[task] >= target_per_task:
                        break
                    try:
                        target = validate_teacher_output(item["task"], text, finish_reason=finish_reason)
                    except (ValueError, TypeError) as exc:
                        rejected.write(
                            json.dumps(
                                {"id": item["id"], "task": item["task"], "reason": str(exc), "text": text},
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        rejected_count += 1
                        continue
                    messages = item["messages"] + [{"role": "assistant", "content": target}]
                    token_count = rendered_token_count(client, messages)
                    if token_count > training_max_length:
                        rejected.write(
                            json.dumps(
                                {
                                    "id": item["id"],
                                    "task": item["task"],
                                    "reason": f"conversation has {token_count} tokens; limit is {training_max_length}",
                                    "text": text,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        rejected_count += 1
                        continue
                    conversation = {
                        "id": item["id"],
                        "task": item["task"],
                        "email_id": item["email_id"],
                        "conversations": messages,
                    }
                    accepted.write(json.dumps(conversation, ensure_ascii=False) + "\n")
                    accepted_count += 1
                    accepted_by_task[task] += 1
                accepted.flush()
                rejected.flush()
                done = min(start + len(batch), len(task_pending))
                print(
                    f"{output_path.stem}/{task}: candidates={done}/{len(task_pending)} "
                    f"accepted={accepted_by_task[task]}/{target_per_task}",
                    flush=True,
                )
                if accepted_by_task[task] >= target_per_task:
                    break
    return {
        "accepted": accepted_count,
        "accepted_by_task": accepted_by_task,
        "rejected": rejected_count,
        "candidate_pool": len(candidates),
        "target_per_task": target_per_task,
    }


def generate_batch(
    client: VLLMClient, batch: list[dict], *, max_tokens: int
) -> tuple[list[str], list[str | None]]:
    prompts = []
    for item in batch:
        rendered = client._render_prompt(item["messages"])
        token_ids = client.tokenizer.encode(rendered, add_special_tokens=False)
        if client.max_model_len is not None:
            token_ids = truncate_token_ids(token_ids, client.max_model_len - max_tokens)
        prompts.append({"prompt_token_ids": token_ids})
    sampling = client.sampling_params_cls(temperature=0.0, max_tokens=max_tokens)
    outputs = client.llm.generate(prompts, sampling, use_tqdm=False)
    texts = [output.outputs[0].text.strip() for output in outputs]
    finish_reasons = [getattr(output.outputs[0], "finish_reason", None) for output in outputs]
    return texts, finish_reasons


def rendered_token_count(client: VLLMClient, messages: list[dict[str, str]]) -> int:
    try:
        rendered = client.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
        )
    except TypeError:
        rendered = client.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    return len(client.tokenizer.encode(rendered, add_special_tokens=False))


def validate_teacher_output(task: str, text: str, *, finish_reason: str | None = None) -> str:
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("empty teacher output")
    if finish_reason == "length":
        raise ValueError("teacher output reached max_tokens")
    if task == "draft_reply":
        if len(cleaned) < 10:
            raise ValueError("reply is too short")
        return cleaned
    data = parse_json_object(cleaned)
    if task == "classify_email":
        if data.get("category") not in VALID_CATEGORIES:
            raise ValueError(f"invalid category: {data.get('category')!r}")
        if data.get("priority") not in {"low", "normal", "high", "urgent"}:
            raise ValueError(f"invalid priority: {data.get('priority')!r}")
        confidence = data.get("confidence")
        if not isinstance(confidence, (int, float)) or not 0 <= float(confidence) <= 1:
            raise ValueError(f"invalid confidence: {confidence!r}")
    elif task == "summarize_email":
        if not isinstance(data.get("summary"), str) or not data["summary"].strip():
            raise ValueError("missing summary")
    elif task == "extract_action_items":
        if not isinstance(data.get("action_items"), list):
            raise ValueError("missing action_items list")
    else:
        raise ValueError(f"unsupported task: {task}")
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def email_prompt(row: dict, *, max_body_chars: int) -> str:
    to_values = row.get("to") or []
    to_text = to_values if isinstance(to_values, str) else ", ".join(str(value) for value in to_values)
    sender = row.get("from", row.get("sender", ""))
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


def stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc


def load_completed_ids(path: Path, *, exclude_tasks: set[str] | None = None) -> set[str]:
    if not path.exists():
        return set()
    exclude_tasks = exclude_tasks or set()
    return {str(row["id"]) for row in read_jsonl(path) if row.get("task") not in exclude_tasks}


def count_jsonl(path: Path) -> int:
    return sum(1 for _ in read_jsonl(path)) if path.exists() else 0


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "train_per_task", "validation_per_task", "max_body_chars", "max_model_len",
        "training_max_length", "batch_size",
    ):
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.oversample_factor < 1:
        raise SystemExit("--oversample-factor must be at least 1")


if __name__ == "__main__":
    main()
