#!/usr/bin/env python3
"""Augment minority email categories with cross-validated synthetic examples.

One configured large model generates an email for a requested category and the
other model labels it blindly. A row is accepted only if that independent label
matches the requested category at the configured confidence threshold.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from email_system.env import load_local_env
from email_system.models.chat_prompts import PROMPT_VERSION, messages_for_task
from email_system.skills.classify import VALID_CATEGORIES
from email_system.skills.json_utils import ModelOutputParseError, parse_json_object
from training.label_multiclass_consensus import (
    DEFAULT_MODELS,
    LABEL_SYSTEM_PROMPT,
    LabelRequestError,
    extract_content,
    labeling_prompt,
    request_label,
)
from training.prepare_lora_classification_data import _email_prompt

load_local_env(ROOT)

TAXONOMY_VERSION = "email-multiclass-synthetic-cross-validated-v1"
DEFAULT_BASE_TRAIN = "data/finetune/multiclass_consensus_v3_maildir/train.jsonl"
DEFAULT_OUTPUT_DIR = "data/finetune/multiclass_consensus_v3_maildir_minority_augmented"
DEFAULT_TARGET_COUNT = 1000

CATEGORY_BRIEFS = {
    "personal_email": "ordinary private conversation, personal update, personal thanks, or family/friend communication",
    "business_email": "direct business transaction, project cooperation, sales correspondence, customer service, recruiting, or interview",
    "internal_email": "internal company announcement, team collaboration, project update, task assignment, or employee feedback",
    "marketing_email": "newsletter, bulk promotion, campaign, discount, or customer-facing event marketing",
    "automated_email": "automated welcome, account event, receipt, order or appointment confirmation, shipping update, or scheduled reminder",
    "legal_formal_email": "legal notice, formal warning, policy violation, overdue-payment warning, or contract delivery/signing/confirmation",
    "educational_email": "course schedule, exam, transcript, assessment, lecture, competition, or educational activity notice",
    "social_email": "greeting, holiday or birthday wish, social invitation, or congratulations",
    "special_purpose_email": "survey, feedback collection, maintenance notice, service interruption, operational alert, or purpose-specific notification",
    "spam": "phishing, scam, credential theft, malware, deceptive message, or unsolicited bulk outreach",
}

GENERATION_SYSTEM_PROMPT = """You create realistic but entirely fictional email training examples.
Produce one email belonging to the requested category. Do not reuse real people, companies,
domains, addresses, phone numbers, account numbers, or legal case details. Do not mention that
the message is synthetic or a training example. Use plausible email style, but keep it concise.
Return only one JSON object with fields subject, from, to, timestamp, and body_text."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-validated synthetic augmentation for minority email classes.")
    parser.add_argument("--base-train-file", default=DEFAULT_BASE_TRAIN)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--base-max-per-category",
        type=int,
        default=0,
        help="Deterministically cap base rows per category before augmentation; 0 keeps every base row.",
    )
    parser.add_argument(
        "--combined-file-name",
        default="train.jsonl",
        help="Combined output JSONL filename inside --output-dir, for example evaluation.jsonl.",
    )
    parser.add_argument(
        "--dataset-role",
        choices=["train", "diagnostic_eval"],
        default="train",
        help="Records whether this output is train data or a teacher-validated diagnostic evaluation set.",
    )
    parser.add_argument("--api-url", default=os.environ.get("EMAILSYSTEM_LABEL_API_URL"))
    parser.add_argument("--api-key-env", default="EMAILSYSTEM_LABEL_API_KEY")
    parser.add_argument("--model", action="append", default=None, help="Exactly two model names; repeat twice.")
    parser.add_argument("--target-count", type=int, default=DEFAULT_TARGET_COUNT, help="Target total rows per selected category.")
    parser.add_argument("--category", action="append", choices=sorted(VALID_CATEGORIES), default=None)
    parser.add_argument("--min-validator-confidence", type=float, default=0.80)
    parser.add_argument("--max-attempts-factor", type=float, default=3.0)
    parser.add_argument("--max-body-chars", type=int, default=4000)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--dry-run", action="store_true", help="Print planned deficits without contacting the API or writing data.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = tuple(args.model or DEFAULT_MODELS)
    _validate_args(args, models)
    base_path = Path(args.base_train_file)
    if not base_path.exists():
        raise SystemExit(f"Missing base train file: {base_path}")

    original_base_rows = list(_read_jsonl(base_path))
    base_rows = _cap_rows_per_category(original_base_rows, args.base_max_per_category, seed=args.seed)
    base_counts = _category_counts(base_rows)
    selected_categories = sorted(set(args.category or VALID_CATEGORIES))
    deficits = {category: max(0, args.target_count - base_counts[category]) for category in selected_categories}
    plan = {
        "base_train_file": str(base_path),
        "uncapped_base_category_counts": dict(sorted(_category_counts(original_base_rows).items())),
        "base_category_counts": dict(sorted(base_counts.items())),
        "target_count": args.target_count,
        "selected_categories": selected_categories,
        "deficits": deficits,
        "planned_new_rows": sum(deficits.values()),
    }
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
        return

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Missing API key environment variable: {args.api_key_env}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "taxonomy_version": TAXONOMY_VERSION,
        "prompt_version": PROMPT_VERSION,
        "base_train_file": str(base_path),
        "base_train_sha256": _file_sha256(base_path),
        "models": list(models),
        "target_count": args.target_count,
        "selected_categories": selected_categories,
        "min_validator_confidence": args.min_validator_confidence,
        "max_attempts_factor": args.max_attempts_factor,
        "max_body_chars": args.max_body_chars,
        "seed": args.seed,
    }
    if args.base_max_per_category:
        config["base_max_per_category"] = args.base_max_per_category
    if args.combined_file_name != "train.jsonl":
        config["combined_file_name"] = args.combined_file_name
    if args.dataset_role != "train":
        config["dataset_role"] = args.dataset_role
    _prepare_config(output_dir / "augmentation_config.json", config)

    accepted_path = output_dir / "augmented_train.jsonl"
    annotations_path = output_dir / "annotations.jsonl"
    accepted_rows = list(_read_jsonl(accepted_path)) if accepted_path.exists() else []
    accepted_fingerprints = {_fingerprint(row) for row in base_rows + accepted_rows}
    accepted_counts = _category_counts(accepted_rows)
    attempts_by_category = Counter(
        item.get("target_category") for item in _read_jsonl(annotations_path)
    ) if annotations_path.exists() else Counter()

    with accepted_path.open("a", encoding="utf-8") as accepted_handle, annotations_path.open("a", encoding="utf-8") as annotation_handle:
        for category in selected_categories:
            needed = max(0, deficits[category] - accepted_counts[category])
            if not needed:
                continue
            max_attempts = max(20, int((deficits[category] * args.max_attempts_factor) + 0.9999))
            print(
                f"{category}: base={base_counts[category]} accepted={accepted_counts[category]} "
                f"needed={needed} prior_attempts={attempts_by_category[category]}/{max_attempts}",
                flush=True,
            )
            while accepted_counts[category] < deficits[category] and attempts_by_category[category] < max_attempts:
                attempt_index = attempts_by_category[category]
                generator, validator = _generator_validator(models, category, attempt_index, args.seed)
                annotation = _augment_once(
                    category=category,
                    generator=generator,
                    validator=validator,
                    api_url=args.api_url,
                    api_key=api_key,
                    min_validator_confidence=args.min_validator_confidence,
                    max_body_chars=args.max_body_chars,
                    timeout=args.timeout,
                    retries=args.retries,
                    known_fingerprints=accepted_fingerprints,
                    attempt_index=attempt_index,
                )
                annotation_handle.write(json.dumps(annotation, ensure_ascii=False) + "\n")
                annotation_handle.flush()
                attempts_by_category[category] += 1
                if annotation["status"] == "accepted":
                    row = _build_training_item(annotation["email"], category, annotation, max_body_chars=args.max_body_chars)
                    accepted_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    accepted_handle.flush()
                    accepted_rows.append(row)
                    accepted_fingerprints.add(_fingerprint(row))
                    accepted_counts[category] += 1
                if attempts_by_category[category] % 10 == 0 or annotation["status"] == "accepted":
                    print(
                        f"{category}: accepted={accepted_counts[category]}/{deficits[category]} "
                        f"attempts={attempts_by_category[category]}/{max_attempts} last={annotation['status']}",
                        flush=True,
                    )

    combined_path = output_dir / args.combined_file_name
    _write_jsonl_atomic(combined_path, base_rows + accepted_rows)
    annotation_rows = list(_read_jsonl(annotations_path))
    _write_jsonl_atomic(output_dir / "rejected.jsonl", (row for row in annotation_rows if row.get("status") != "accepted"))
    final_counts = _category_counts(base_rows + accepted_rows)
    manifest = {
        **config,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_category_counts": dict(sorted(base_counts.items())),
        "synthetic_category_counts": dict(sorted(_category_counts(accepted_rows).items())),
        "combined_category_counts": dict(sorted(final_counts.items())),
        "accepted_synthetic_rows": len(accepted_rows),
        "annotation_status_counts": dict(sorted(Counter(row.get("status") for row in annotation_rows).items())),
        "combined_file": args.combined_file_name,
        "augmented_train_file": "augmented_train.jsonl",
        "dataset_role": args.dataset_role,
        "evaluation_warning": (
            "Teacher-validated synthetic diagnostic data; do not use as the primary real-world quality metric."
            if args.dataset_role == "diagnostic_eval"
            else "Synthetic examples are train-only; keep validation data unchanged."
        ),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


def _augment_once(
    *,
    category: str,
    generator: str,
    validator: str,
    api_url: str,
    api_key: str,
    min_validator_confidence: float,
    max_body_chars: int,
    timeout: float,
    retries: int,
    known_fingerprints: set[str],
    attempt_index: int,
) -> dict:
    common = {
        "target_category": category,
        "generator_model": generator,
        "validator_model": validator,
        "attempt_index": attempt_index,
        "taxonomy_version": TAXONOMY_VERSION,
    }
    try:
        email = _request_generated_email(
            api_url, api_key, generator, _generation_prompt(category), timeout=timeout, retries=retries
        )
        fingerprint = _fingerprint(email)
        if fingerprint in known_fingerprints:
            return {**common, "status": "duplicate", "reason": "content fingerprint already exists", "email": email}
        validator_label = request_label(
            api_url,
            api_key,
            validator,
            labeling_prompt(email, max_body_chars=max_body_chars),
            timeout=timeout,
            retries=retries,
        )
        if validator_label["category"] != category:
            return {
                **common,
                "status": "rejected",
                "reason": f"validator category={validator_label['category']}",
                "validator_label": validator_label,
                "email": email,
            }
        if validator_label["confidence"] < min_validator_confidence:
            return {
                **common,
                "status": "rejected",
                "reason": f"validator confidence={validator_label['confidence']:.3f} below threshold",
                "validator_label": validator_label,
                "email": email,
            }
        return {**common, "status": "accepted", "validator_label": validator_label, "email": email}
    except LabelRequestError as exc:
        return {**common, "status": "error", "reason": str(exc), "raw_preview": exc.raw_preview}
    except Exception as exc:
        return {**common, "status": "error", "reason": f"{type(exc).__name__}: {exc}"}


def _request_generated_email(
    api_url: str,
    api_key: str,
    model: str,
    prompt: str,
    *,
    timeout: float,
    retries: int,
) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": GENERATION_SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        "stream": False,
        "temperature": 0.7,
        "max_tokens": 900,
        "response_format": {"type": "json_object"},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        api_url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    last_error: Exception | None = None
    last_raw = ""
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                content = extract_content(json.loads(response.read().decode("utf-8")))
            last_raw = content
            return _normalize_generated_email(parse_json_object(content))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError, ValueError, ModelOutputParseError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(min(2**attempt, 8))
    raise LabelRequestError(f"generation failed after {retries} attempts: {last_error}", raw_preview=last_raw)


def _normalize_generated_email(value: dict) -> dict:
    subject = _clean_text(value.get("subject"), limit=240)
    sender = _clean_text(value.get("from"), limit=240)
    body = _clean_text(value.get("body_text", value.get("body")), limit=8000)
    if len(subject) < 3 or len(body) < 80:
        raise ValueError("generated email must include a subject and at least 80 body characters")
    to_value = value.get("to", [])
    to = [_clean_text(item, limit=240) for item in to_value] if isinstance(to_value, list) else [_clean_text(to_value, limit=240)]
    return {
        "email_id": "synthetic:" + hashlib.sha256((subject + "\n" + body).encode("utf-8")).hexdigest()[:24],
        "thread_id": None,
        "subject": subject,
        "from": sender,
        "to": [item for item in to if item],
        "cc": [],
        "timestamp": _clean_text(value.get("timestamp"), limit=64),
        "body_text": body,
        "source": "synthetic_dual_model_cross_validated",
        "labels": {},
    }


def _build_training_item(email: dict, category: str, annotation: dict, *, max_body_chars: int) -> dict:
    confidence = float(annotation["validator_label"]["confidence"])
    prompt = _email_prompt(email, max_body_chars=max_body_chars)
    messages = messages_for_task(prompt, task="classify_email")
    messages.append(
        {"role": "assistant", "content": json.dumps({"category": category, "priority": "normal", "confidence": confidence}, separators=(",", ":"))}
    )
    return {
        **email,
        "body_text": str(email["body_text"])[:max_body_chars],
        "labels": {"category": category, "category_source": "synthetic_cross_validated"},
        "category_label": category,
        "label_source": "synthetic_cross_validated",
        "teacher_models": [annotation["generator_model"], annotation["validator_model"]],
        "synthetic_provenance": {
            "generator_model": annotation["generator_model"],
            "validator_model": annotation["validator_model"],
            "validator_confidence": confidence,
            "target_category": category,
        },
        "messages": messages,
    }


def _generation_prompt(category: str) -> str:
    return (
        f"Create one realistic fictional email whose intended category is {category}: {CATEGORY_BRIEFS[category]}.\n"
        "Use one concrete subtype from the category definition, vary the scenario and wording, and avoid generic filler.\n"
        "Return only JSON with subject, from, to, timestamp, body_text."
    )


def _generator_validator(models: tuple[str, str], category: str, attempt_index: int, seed: int) -> tuple[str, str]:
    offset = int(hashlib.sha256(f"{seed}:{category}:{attempt_index}".encode("utf-8")).hexdigest(), 16) % 2
    return (models[offset], models[1 - offset])


def _category_counts(rows: Iterable[dict]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        category = row.get("category_label") or (row.get("labels") or {}).get("category")
        if category in VALID_CATEGORIES:
            counts[str(category)] += 1
    return counts


def _cap_rows_per_category(rows: list[dict], max_per_category: int, *, seed: int) -> list[dict]:
    if max_per_category <= 0:
        return rows
    groups: dict[str, list[dict]] = {}
    for row in rows:
        category = row.get("category_label") or (row.get("labels") or {}).get("category")
        if category in VALID_CATEGORIES:
            groups.setdefault(str(category), []).append(row)
    selected = []
    for category in sorted(groups):
        ordered = sorted(
            groups[category],
            key=lambda row: hashlib.sha256(f"{seed}:{row.get('email_id', _fingerprint(row))}".encode("utf-8")).hexdigest(),
        )
        selected.extend(ordered[:max_per_category])
    return selected


def _fingerprint(row: dict) -> str:
    value = "\n".join([str(row.get("subject", "")), str(row.get("body_text", ""))])
    normalized = re.sub(r"\s+", " ", value.lower()).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _clean_text(value: object, *, limit: int) -> str:
    text = str(value or "").replace("\x00", " ")
    return re.sub(r"\s+", " ", text).strip()[:limit]


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc


def _write_jsonl_atomic(path: Path, rows: Iterable[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _prepare_config(path: Path, config: dict) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != config:
            raise SystemExit(f"Existing augmentation configuration differs: {path}. Use a new --output-dir.")
        return
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_args(args: argparse.Namespace, models: tuple[str, ...]) -> None:
    if not args.dry_run and not args.api_url:
        raise SystemExit("Missing --api-url or EMAILSYSTEM_LABEL_API_URL")
    if len(models) != 2 or models[0] == models[1]:
        raise SystemExit("Provide exactly two distinct --model values")
    if args.target_count <= 0 or args.max_body_chars <= 0 or args.retries <= 0 or args.base_max_per_category < 0:
        raise SystemExit("--target-count, --max-body-chars, and --retries must be positive")
    if args.timeout <= 0 or args.max_attempts_factor <= 0:
        raise SystemExit("--timeout and --max-attempts-factor must be positive")
    if not 0 <= args.min_validator_confidence <= 1:
        raise SystemExit("--min-validator-confidence must be between 0 and 1")
    if Path(args.combined_file_name).name != args.combined_file_name or not args.combined_file_name.endswith(".jsonl"):
        raise SystemExit("--combined-file-name must be a JSONL filename without directory components")


if __name__ == "__main__":
    main()
