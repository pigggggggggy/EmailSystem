#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from email_system.models.chat_prompts import PROMPT_VERSION, messages_for_task
from email_system.skills.classify import VALID_CATEGORIES
from email_system.skills.json_utils import parse_json_object
from training.prepare_lora_classification_data import _email_prompt

DEFAULT_INPUT_DIRS = ("data/processed/spam_benchmark", "data/processed/phishing_benchmark", "data/processed/maildir_benchmark")
DEFAULT_INPUT_WEIGHTS = (1.0, 1.0, 3.0)
DEFAULT_MODELS = ("gemma4-26b", "qwen3.6-27b")
TAXONOMY_VERSION = "email-multiclass-consensus-v3"
LABEL_SYSTEM_PROMPT = """You are a strict email classification annotator.
Choose exactly one category using these precedence-aware definitions:
- personal_email: private conversations among friends, family, or acquaintances, including daily life, travel, personal feelings, and personal thanks.
- business_email: business transactions, negotiations, project cooperation, sales conversations, customer service, complaints, after-sales support, recruiting, and interviews.
- internal_email: internal company announcements, policy changes, team collaboration, project updates, task assignments, meeting coordination, and employee feedback.
- marketing_email: newsletters, company or industry news digests, promotions, discounts, product campaigns, and customer-facing event invitations.
- automated_email: automated welcome messages, order or appointment confirmations, account events, receipts, shipping updates, and scheduled reminders.
- legal_formal_email: legal notices, formal warnings, policy violations, overdue-payment warnings, and contract delivery, signing, or confirmation.
- educational_email: course schedules, exams, transcripts, assessment reports, lectures, competitions, and educational activity notices.
- social_email: greetings, holiday or birthday wishes, social invitations, and congratulations.
- special_purpose_email: surveys, feedback collection, maintenance notices, service interruptions, operational alerts, and other purpose-specific notifications.
- spam: phishing, scams, credential theft, malware, deceptive mail, unsolicited bulk messages, or unwanted outreach. Spam takes precedence over every other category.
Distinguish personal_email from social_email: ordinary private conversation is personal_email; greetings, invitations, and congratulations are social_email.
Distinguish business_email from marketing_email: direct business, sales, service, or recruiting correspondence is business_email; bulk promotion, newsletters, and campaign invitations are marketing_email.
Distinguish automated_email from special_purpose_email: transactional lifecycle messages are automated_email; surveys, maintenance, outages, and operational notices are special_purpose_email.
There is no other category. For ambiguous legitimate mail, select the closest category and lower confidence.
Return only one JSON object: {"category":"...","confidence":0.0}.
category must be one of personal_email, business_email, internal_email, marketing_email, automated_email, legal_formal_email, educational_email, social_email, special_purpose_email, spam.
confidence must be a number from 0 to 1. Do not include explanations."""


class LabelRequestError(RuntimeError):
    def __init__(self, message: str, raw_preview: str = "") -> None:
        super().__init__(message)
        self.raw_preview = raw_preview[:1000]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create multiclass LoRA data using two-model consensus labels.")
    parser.add_argument("--input-dir", action="append", default=None)
    parser.add_argument("--input-weight", action="append", type=float, default=None, help="Relative source quota; repeat once per --input-dir.")
    parser.add_argument("--output-dir", default="data/finetune/multiclass_consensus")
    parser.add_argument("--api-url", default=os.environ.get("EMAILSYSTEM_LABEL_API_URL"))
    parser.add_argument("--api-key-env", default="EMAILSYSTEM_LABEL_API_KEY")
    parser.add_argument("--model", action="append", default=None, help="Exactly two model names; repeat twice.")
    parser.add_argument("--train-limit", type=int, default=10000)
    parser.add_argument("--validation-limit", type=int, default=1000)
    parser.add_argument("--max-body-chars", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260703)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = tuple(args.model or DEFAULT_MODELS)
    validate_args(args, models)
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Missing API key environment variable: {args.api_key_env}")

    input_dirs = [Path(value) for value in (args.input_dir or DEFAULT_INPUT_DIRS)]
    input_weights = tuple(args.input_weight or (DEFAULT_INPUT_WEIGHTS if args.input_dir is None else [1.0] * len(input_dirs)))
    validate_input_weights(input_dirs, input_weights)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "taxonomy_version": TAXONOMY_VERSION,
        "prompt_version": PROMPT_VERSION,
        "source_dirs": [str(path) for path in input_dirs],
        "source_weights": list(input_weights),
        "models": list(models),
        "train_limit": args.train_limit,
        "validation_limit": args.validation_limit,
        "max_body_chars": args.max_body_chars,
        "seed": args.seed,
    }
    prepare_config(output_dir / "labeling_config.json", config)

    split_metrics = {}
    for split, limit in (("train", args.train_limit), ("validation", args.validation_limit)):
        rows = weighted_sample_input_dirs(input_dirs, input_weights, split=split, limit=limit, seed=args.seed + (split == "validation"))
        split_metrics[split] = label_split(
            rows,
            split=split,
            output_dir=output_dir,
            api_url=args.api_url,
            api_key=api_key,
            models=models,
            max_body_chars=args.max_body_chars,
            batch_size=args.batch_size,
            workers=args.workers,
            timeout=args.timeout,
            retries=args.retries,
        )

    manifest = {
        **config,
        "task": "multiclass_dual_model_consensus",
        "categories": sorted(VALID_CATEGORIES),
        "splits": split_metrics,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


def label_split(
    rows: list[dict],
    *,
    split: str,
    output_dir: Path,
    api_url: str,
    api_key: str,
    models: tuple[str, str],
    max_body_chars: int,
    batch_size: int,
    workers: int,
    timeout: float,
    retries: int,
) -> dict:
    annotations_path = output_dir / f"{split}_annotations.jsonl"
    existing = latest_annotations(annotations_path)
    completed = {row_id for row_id, item in existing.items() if item.get("status") in {"accepted", "disagreement"}}
    pending = [row for row in rows if row_id(row) not in completed]
    print(f"{split}: selected={len(rows)} completed={len(completed)} pending={len(pending)}", flush=True)

    with annotations_path.open("a", encoding="utf-8") as handle:
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            annotations = annotate_batch(
                batch,
                api_url=api_url,
                api_key=api_key,
                models=models,
                max_body_chars=max_body_chars,
                workers=workers,
                timeout=timeout,
                retries=retries,
            )
            for annotation in annotations:
                handle.write(json.dumps(annotation, ensure_ascii=False) + "\n")
            handle.flush()
            done = min(start + len(batch), len(pending))
            counts = Counter(item["status"] for item in annotations)
            print(f"{split}: {done}/{len(pending)} batch={dict(counts)}", flush=True)

    annotations = latest_annotations(annotations_path)
    selected_ids = {row_id(row) for row in rows}
    accepted_rows = []
    disagreements = []
    errors = []
    category_counts = Counter()
    for row in rows:
        identifier = row_id(row)
        annotation = annotations.get(identifier)
        if not annotation:
            continue
        status = annotation.get("status")
        if status == "accepted":
            item = build_training_item(row, annotation, max_body_chars=max_body_chars)
            accepted_rows.append(item)
            category_counts[item["category_label"]] += 1
        elif status == "disagreement":
            disagreements.append(annotation)
        elif status == "error":
            errors.append(annotation)

    write_jsonl_atomic(output_dir / f"{split}.jsonl", accepted_rows)
    write_jsonl_atomic(output_dir / f"{split}_disagreements.jsonl", disagreements)
    write_jsonl_atomic(output_dir / f"{split}_errors.jsonl", errors)
    return {
        "selected": len(rows),
        "selected_sources": dict(sorted(Counter(str(row.get("source", "unknown")) for row in rows).items())),
        "accepted": len(accepted_rows),
        "agreement_rate": len(accepted_rows) / len(rows) if rows else 0.0,
        "disagreements": len(disagreements),
        "errors": len(errors),
        "category_counts": dict(sorted(category_counts.items())),
        "completed_annotations": sum(identifier in selected_ids for identifier in annotations),
    }


def annotate_batch(
    rows: list[dict],
    *,
    api_url: str,
    api_key: str,
    models: tuple[str, str],
    max_body_chars: int,
    workers: int,
    timeout: float,
    retries: int,
) -> list[dict]:
    results: list[dict[str, dict]] = [dict() for _ in rows]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for index, row in enumerate(rows):
            prompt = labeling_prompt(row, max_body_chars=max_body_chars)
            for model in models:
                future = executor.submit(
                    request_label,
                    api_url,
                    api_key,
                    model,
                    prompt,
                    timeout=timeout,
                    retries=retries,
                )
                futures[future] = (index, model)
        for future in as_completed(futures):
            index, model = futures[future]
            try:
                results[index][model] = future.result()
            except LabelRequestError as exc:
                results[index][model] = {
                    "error": f"{type(exc).__name__}: {exc}",
                    "raw_preview": exc.raw_preview,
                }
            except Exception as exc:
                results[index][model] = {"error": f"{type(exc).__name__}: {exc}"}

    annotations = []
    for row, labels in zip(rows, results):
        decision = consensus_decision(labels, models)
        annotations.append(
            {
                "email_id": row_id(row),
                "source": row.get("source"),
                "original_labels": row.get("labels", {}),
                "status": decision["status"],
                "consensus_category": decision.get("category"),
                "consensus_confidence": decision.get("confidence"),
                "model_labels": labels,
                "reason": decision.get("reason"),
                "taxonomy_version": TAXONOMY_VERSION,
            }
        )
    return annotations


def request_label(
    api_url: str,
    api_key: str,
    model: str,
    prompt: str,
    *,
    timeout: float,
    retries: int,
) -> dict:
    payload = build_request_payload(model, prompt)
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        api_url,
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    last_error = None
    last_raw = ""
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                response_data = json.loads(response.read().decode("utf-8"))
            content = extract_content(response_data)
            last_raw = content
            label = validate_label(parse_json_object(content))
            return {**label, "raw": content}
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(min(2**attempt, 8))
    raise LabelRequestError(
        f"request failed after {retries} attempts: {last_error}",
        raw_preview=last_raw,
    )


def build_request_payload(model: str, prompt: str) -> dict:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": LABEL_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "temperature": 0.0,
        "max_tokens": 256,
        "response_format": {"type": "json_object"},
        "chat_template_kwargs": {"enable_thinking": False},
    }


def extract_content(response: dict) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("response has no choices")
    message = choices[0].get("message", {})
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content
    if isinstance(content, list):
        text = "".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
        if text.strip():
            return text
    for key in ("reasoning_content", "reasoning"):
        reasoning = message.get(key)
        if isinstance(reasoning, str) and reasoning.strip():
            return reasoning
    raise ValueError("response has no text or reasoning content")

def validate_label(value: dict) -> dict:
    category = value.get("category")
    confidence = value.get("confidence")
    if category not in VALID_CATEGORIES:
        raise ValueError(f"invalid category: {category!r}")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise ValueError(f"invalid confidence: {confidence!r}")
    return {"category": str(category), "confidence": float(confidence)}


def consensus_decision(labels: dict[str, dict], models: tuple[str, str]) -> dict:
    missing = [model for model in models if "error" in labels.get(model, {}) or model not in labels]
    if missing:
        return {"status": "error", "reason": f"model errors: {', '.join(missing)}"}
    first, second = (labels[model] for model in models)
    if first["category"] != second["category"]:
        return {
            "status": "disagreement",
            "reason": f"category disagreement: {first['category']} != {second['category']}",
        }
    return {
        "status": "accepted",
        "category": first["category"],
        "confidence": min(float(first["confidence"]), float(second["confidence"])),
    }


def build_training_item(row: dict, annotation: dict, *, max_body_chars: int) -> dict:
    category = str(annotation["consensus_category"])
    confidence = float(annotation["consensus_confidence"])
    prompt = _email_prompt(row, max_body_chars=max_body_chars)
    messages = messages_for_task(prompt, task="classify_email")
    target = json.dumps(
        {"category": category, "priority": "normal", "confidence": confidence},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    messages.append({"role": "assistant", "content": target})
    labels = dict(row.get("labels") or {})
    labels["category"] = category
    labels["category_source"] = "dual_model_consensus"
    return {
        "email_id": row_id(row),
        "thread_id": row.get("thread_id", row_id(row)),
        "subject": row.get("subject", ""),
        "from": row.get("from", row.get("sender", "")),
        "to": row.get("to", []),
        "cc": row.get("cc", []),
        "timestamp": row.get("timestamp"),
        "body_text": str(row.get("body_text", ""))[:max_body_chars],
        "source": row.get("source"),
        "labels": labels,
        "category_label": category,
        "label_source": "dual_model_consensus",
        "teacher_models": list(annotation.get("model_labels", {}).keys()),
        "original_labels": row.get("labels", {}),
        "messages": messages,
    }


def labeling_prompt(row: dict, *, max_body_chars: int) -> str:
    return "Classify this email.\n\n" + _email_prompt(row, max_body_chars=max_body_chars)


def weighted_sample_input_dirs(
    input_dirs: list[Path],
    weights: tuple[float, ...],
    *,
    split: str,
    limit: int,
    seed: int,
) -> list[dict]:
    source_rows = [collect_rows([input_dir], split) for input_dir in input_dirs]
    allocations = _weighted_allocations([len(rows) for rows in source_rows], weights, limit)
    selected = []
    for index, (rows, allocation) in enumerate(zip(source_rows, allocations)):
        selected.extend(stable_sample(rows, limit=allocation, seed=seed + index))
    return sorted(selected, key=lambda row: stable_key(seed, row_id(row)))


def _weighted_allocations(sizes: list[int], weights: tuple[float, ...], limit: int) -> list[int]:
    target = sum(sizes) if limit <= 0 else min(limit, sum(sizes))
    allocations = [0] * len(sizes)
    for _ in range(target):
        eligible = [index for index, size in enumerate(sizes) if allocations[index] < size]
        if not eligible:
            break
        selected = min(eligible, key=lambda index: ((allocations[index] + 1) / weights[index], index))
        allocations[selected] += 1
    return allocations


def validate_input_weights(input_dirs: list[Path], weights: tuple[float, ...]) -> None:
    if len(input_dirs) != len(weights):
        raise SystemExit("Provide exactly one --input-weight per --input-dir")
    if any(weight <= 0 for weight in weights):
        raise SystemExit("--input-weight values must be positive")


def collect_rows(input_dirs: list[Path], split: str) -> list[dict]:
    rows_by_id = {}
    for input_dir in input_dirs:
        path = input_dir / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"missing split file: {path}")
        for index, row in enumerate(read_jsonl(path)):
            identifier = str(row.get("email_id") or f"{path}:{index}")
            rows_by_id.setdefault(identifier, row)
    return list(rows_by_id.values())


def stable_sample(rows: Iterable[dict], *, limit: int, seed: int) -> list[dict]:
    ordered = sorted(rows, key=lambda row: stable_key(seed, row_id(row)))
    return ordered[:limit] if limit > 0 else ordered


def stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def row_id(row: dict) -> str:
    return str(row.get("email_id", ""))


def prepare_config(path: Path, config: dict) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != config:
            raise SystemExit(f"Existing labeling configuration differs: {path}. Use a new --output-dir.")
        return
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def latest_annotations(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {str(item["email_id"]): item for item in read_jsonl(path)}


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc


def write_jsonl_atomic(path: Path, rows: Iterable[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def validate_args(args: argparse.Namespace, models: tuple[str, ...]) -> None:
    if not args.api_url:
        raise SystemExit("Missing --api-url or EMAILSYSTEM_LABEL_API_URL")
    if len(models) != 2 or models[0] == models[1]:
        raise SystemExit("Provide exactly two distinct --model values")
    for name in ("train_limit", "validation_limit", "max_body_chars", "batch_size", "workers", "retries"):
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.timeout <= 0:
        raise SystemExit("--timeout must be positive")


if __name__ == "__main__":
    main()
