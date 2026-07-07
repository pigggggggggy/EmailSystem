#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from email_system.models.chat_prompts import messages_for_task
from email_system.skills.json_utils import fallback_summary_from_text
from training.prepare_lora_classification_data import _email_prompt

TASKS = ("classify_email", "summarize_email", "extract_action_items", "draft_reply")
DEFAULT_TASK_WEIGHTS = {
    "classify_email": 10,
    "summarize_email": 4,
    "extract_action_items": 2,
    "draft_reply": 4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare chat-format multitask LoRA data for the email agent.")
    parser.add_argument("--input-dir", default="data/finetune/multiclass_consensus_v3_maildir")
    parser.add_argument("--output-dir", default="data/finetune/multitask_lora")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--validation-split", default="validation")
    parser.add_argument("--max-body-chars", type=int, default=6000)
    parser.add_argument("--task-weight", action="append", default=None, help="Task inclusion weight from 0 to 10, e.g. summarize_email=4 means about 40% of rows. Repeat to override defaults.")
    parser.add_argument("--seed", type=int, default=20260707)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    weights = _parse_task_weights(args.task_weight)

    train_rows = list(_read_jsonl(input_dir / f"{args.train_split}.jsonl"))
    validation_rows = list(_read_jsonl(input_dir / f"{args.validation_split}.jsonl"))
    if not train_rows:
        raise SystemExit(f"empty train split: {input_dir / f'{args.train_split}.jsonl'}")
    if not validation_rows:
        raise SystemExit(f"empty validation split: {input_dir / f'{args.validation_split}.jsonl'}")

    train_counts = _write_split(
        output_dir / "train.jsonl",
        train_rows,
        weights=weights,
        max_body_chars=args.max_body_chars,
        seed=args.seed,
    )
    validation_counts = _write_split(
        output_dir / "validation.jsonl",
        validation_rows,
        weights=weights,
        max_body_chars=args.max_body_chars,
        seed=args.seed + 1,
    )
    manifest = {
        "task": "email_agent_multitask_lora",
        "source_dir": str(input_dir),
        "output_dir": str(output_dir),
        "max_body_chars": args.max_body_chars,
        "task_weights": weights,
        "train_records": sum(train_counts.values()),
        "validation_records": sum(validation_counts.values()),
        "train_tasks": dict(sorted(train_counts.items())),
        "validation_tasks": dict(sorted(validation_counts.items())),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def _write_split(
    path: Path,
    rows: list[dict],
    *,
    weights: dict[str, int],
    max_body_chars: int,
    seed: int,
) -> Counter:
    counts: Counter = Counter()
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            for task in _tasks_for_row(row, weights, seed=seed):
                item = build_multitask_item(row, task, max_body_chars=max_body_chars)
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                counts[task] += 1
    return counts


def build_multitask_item(row: dict, task: str, *, max_body_chars: int) -> dict:
    if task not in TASKS:
        raise ValueError(f"unsupported task: {task}")
    prompt = _email_prompt(row, max_body_chars=max_body_chars)
    messages = messages_for_task(prompt, task=task)
    messages.append({"role": "assistant", "content": _target_for_task(row, task)})
    labels = row.get("labels") or {}
    category = labels.get("category") or row.get("category_label")
    return {
        "email_id": row.get("email_id"),
        "source": row.get("source"),
        "task": task,
        "category_label": category,
        "messages": messages,
    }


def _target_for_task(row: dict, task: str) -> str:
    if task == "classify_email":
        category = (row.get("labels") or {}).get("category") or row.get("category_label")
        confidence = float(row.get("consensus_confidence") or row.get("confidence") or 0.85)
        target = {"category": category, "priority": _priority(row), "confidence": max(0.0, min(1.0, confidence))}
        return json.dumps(target, ensure_ascii=False, separators=(",", ":"))
    if task == "summarize_email":
        summary = _summary(row)
        return json.dumps({"summary": summary, "confidence": 0.72}, ensure_ascii=False, separators=(",", ":"))
    if task == "extract_action_items":
        return json.dumps({"action_items": _action_items(row)}, ensure_ascii=False, separators=(",", ":"))
    if task == "draft_reply":
        return _reply_draft(row)
    raise ValueError(f"unsupported task: {task}")


def _summary(row: dict) -> str:
    subject = str(row.get("subject") or "").strip()
    body = str(row.get("body_text") or "")
    if subject:
        return f"{subject}：{fallback_summary_from_text(body, max_chars=90)}"
    return fallback_summary_from_text(body, max_chars=120)


def _action_items(row: dict) -> list[dict]:
    text = f"{row.get('subject', '')}\n{row.get('body_text', '')}".lower()
    if re.search(r"\b(please|need|review|send|check|confirm|reply|respond|schedule|approve)\b|请|需要|确认|检查|回复|安排|审批", text):
        return [{"owner": None, "task": "跟进邮件中请求确认或处理的事项", "due": None}]
    return []


def _reply_draft(row: dict) -> str:
    category = (row.get("labels") or {}).get("category") or row.get("category_label")
    subject = str(row.get("subject") or "").strip()
    if category == "spam":
        return "您好，这封邮件看起来存在广告、诈骗或钓鱼风险。建议不要回复、不要点击链接，也不要提供任何个人或账户信息。"
    if category == "marketing_email":
        return "您好，感谢您的来信。相关信息我已收到，如后续有明确需求会再与您联系。谢谢。"
    if category == "personal_email":
        return "您好，邮件我收到了。谢谢你告诉我这些情况，我会认真看一下，之后再和你进一步沟通。"
    if category == "social_email":
        return "您好，谢谢你的邀请和问候。我已经收到相关信息，确认安排后会再回复你。"
    if category == "automated_email":
        return "您好，相关确认信息我已收到。若后续需要进一步处理，我会按照邮件中的正式渠道核对后再操作。"
    if category == "legal_formal_email":
        return "您好，相关正式文件已收到。我会先核对内容和依据，必要时再通过正式渠道进一步确认。"
    if subject:
        return f"您好，关于“{subject}”这封邮件我已收到。我会核对相关信息，并在确认后尽快回复下一步安排。"
    return "您好，邮件已收到。我会先核对相关信息，并在确认后尽快回复下一步安排。"


def _priority(row: dict) -> str:
    text = f"{row.get('subject', '')}\n{row.get('body_text', '')}".lower()
    if re.search(r"\b(urgent|asap|immediately|deadline|overdue)\b|紧急|立即|尽快|截止|逾期", text):
        return "urgent"
    return "normal"


def _tasks_for_row(row: dict, weights: dict[str, int], *, seed: int) -> list[str]:
    selected = []
    for task, weight in weights.items():
        if weight <= 0:
            continue
        if weight >= 10 or _stable_bucket(seed, str(row.get("email_id", "")), task) < weight:
            selected.append(task)
    return selected or ["classify_email"]


def _stable_bucket(seed: int, email_id: str, task: str) -> int:
    digest = hashlib.sha256(f"{seed}:{email_id}:{task}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 10


def _parse_task_weights(values: list[str] | None) -> dict[str, int]:
    weights = dict(DEFAULT_TASK_WEIGHTS)
    for value in values or []:
        if "=" not in value:
            raise SystemExit(f"invalid --task-weight: {value!r}; expected task=0..10")
        task, raw_weight = value.split("=", 1)
        if task not in TASKS:
            raise SystemExit(f"unsupported task in --task-weight: {task}")
        weight = int(raw_weight)
        if not 0 <= weight <= 10:
            raise SystemExit("--task-weight values must be between 0 and 10")
        weights[task] = weight
    return weights


def _read_jsonl(path: Path) -> Iterable[dict]:
    if not path.exists():
        raise FileNotFoundError(f"missing split file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc


if __name__ == "__main__":
    main()
