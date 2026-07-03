from __future__ import annotations

import hashlib
import math
import time
from collections import Counter
from typing import Any, Callable, Iterable

from email_system.models import LLMClient
from email_system.schemas import Email
from email_system.skills import ClassifyEmailSkill, DraftReplySkill, ExtractActionItemsSkill, SummarizeEmailSkill
from email_system.skills.classify import LOW_CONFIDENCE_THRESHOLD, VALID_CATEGORIES

from .metrics import classification_metrics


TASKS = ("classify_email", "summarize_email", "extract_action_items", "draft_reply")


def resolve_quality_mode(rows: list[dict], requested: str = "auto") -> str:
    if requested not in {"auto", "binary", "multiclass"}:
        raise ValueError(f"unsupported quality mode: {requested}")
    if requested != "auto":
        mode = requested
    else:
        mode = "multiclass" if rows and all(_gold_category(row) in VALID_CATEGORIES for row in rows) else "binary"
    if mode == "multiclass" and any(_gold_category(row) not in VALID_CATEGORIES for row in rows):
        raise ValueError("multiclass quality mode requires labels.category on every row")
    return mode


def run_classification_quality(
    llm: LLMClient,
    rows: list[dict],
    *,
    quality_mode: str = "auto",
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[list[dict], dict]:
    mode = resolve_quality_mode(rows, quality_mode)
    skill = ClassifyEmailSkill()
    predictions = []
    for index, row in enumerate(rows, start=1):
        email = Email.from_dict(row)
        output = skill.run(email, {}, llm)
        predicted_category = str(output.get("category", "other"))
        invalid = bool(output.get("parse_error")) or predicted_category not in VALID_CATEGORIES
        confidence = float(output.get("confidence", 0.0))
        low_confidence = bool(output.get("low_confidence", confidence < LOW_CONFIDENCE_THRESHOLD))
        if mode == "multiclass":
            gold_label = _gold_category(row)
            predicted_label = predicted_category if not invalid else _wrong_category(gold_label)
        else:
            gold_label = _gold_spam_label(row)
            predicted_label = (
                ("spam" if predicted_category == "spam" else "ham")
                if not invalid
                else ("ham" if gold_label == "spam" else "spam")
            )
        prediction = {
            "email_id": email.email_id,
            "quality_mode": mode,
            "gold_label": gold_label,
            "predicted_label": predicted_label,
            "predicted_category": predicted_category,
            "confidence": confidence,
            "low_confidence": low_confidence,
            "accepted_prediction": not invalid and not low_confidence,
            "parse_error": output.get("parse_error"),
            "valid_category": not invalid,
            "usage": output.get("usage", {}),
        }
        if mode == "multiclass":
            prediction.update(gold_category=gold_label, scored_category=predicted_label)
        else:
            prediction.update(gold_spam_label=gold_label, predicted_spam_label=predicted_label)
        predictions.append(prediction)
        if progress_callback is not None:
            progress_callback(index, len(rows))

    metrics = classification_quality_metrics(predictions)
    return predictions, metrics


def classification_quality_metrics(predictions: list[dict]) -> dict:
    mode = predictions[0].get("quality_mode", "binary") if predictions else "binary"
    y_true = [row.get("gold_label", row.get("gold_spam_label")) for row in predictions]
    y_pred = [row.get("predicted_label", row.get("predicted_spam_label")) for row in predictions]
    metrics = classification_metrics(y_true, y_pred)
    metrics["quality_mode"] = mode
    metrics["confusion_matrix"] = (
        _multiclass_confusion(y_true, y_pred) if mode == "multiclass" else _binary_confusion(y_true, y_pred)
    )
    metrics["spam_precision"] = metrics["per_class"].get("spam", {}).get("precision", 0.0)
    metrics["spam_recall"] = metrics["per_class"].get("spam", {}).get("recall", 0.0)
    metrics["parse_success_rate"] = _success_rate(row.get("parse_error") for row in predictions)
    metrics["valid_category_rate"] = _success_rate(not row["valid_category"] for row in predictions)
    accepted = [row for row in predictions if row.get("accepted_prediction", False)]
    metrics["confidence_threshold"] = LOW_CONFIDENCE_THRESHOLD
    metrics["low_confidence_count"] = sum(row.get("low_confidence", False) for row in predictions)
    metrics["low_confidence_rate"] = metrics["low_confidence_count"] / len(predictions) if predictions else 0.0
    metrics["accepted_coverage"] = len(accepted) / len(predictions) if predictions else 0.0
    metrics["accepted_accuracy"] = (
        sum(
            row.get("gold_label", row.get("gold_spam_label"))
            == row.get("predicted_label", row.get("predicted_spam_label"))
            for row in accepted
        )
        / len(accepted)
        if accepted
        else 0.0
    )
    metrics["predicted_categories"] = dict(sorted(Counter(row["predicted_category"] for row in predictions).items()))
    metrics["samples"] = len(predictions)
    return metrics

def run_task_speed(
    llm: LLMClient,
    rows: list[dict],
    *,
    tasks: Iterable[str] = TASKS,
    warmup: int = 2,
    show_progress: bool = False,
) -> tuple[list[dict], dict]:
    runners = _task_runners()
    unknown = sorted(set(tasks) - set(runners))
    if unknown:
        raise ValueError(f"unsupported speed tasks: {', '.join(unknown)}")

    task_names = list(tasks)
    progress = _speed_progress(len(rows) * len(task_names)) if show_progress else None
    samples = []
    for task in task_names:
        runner = runners[task]
        if progress is not None:
            progress.set_postfix_str(task)
        for row in rows[:warmup]:
            runner(Email.from_dict(row), llm)
        for row in rows:
            email = Email.from_dict(row)
            wall_start = time.perf_counter()
            output = runner(email, llm)
            wall_ms = (time.perf_counter() - wall_start) * 1000
            usage = output.get("usage", {})
            samples.append(
                {
                    "task": task,
                    "email_id": email.email_id,
                    "body_chars": len(email.body_text),
                    "model_latency_ms": float(usage.get("latency_ms", 0.0)),
                    "wall_latency_ms": wall_ms,
                    "input_tokens": int(usage.get("input_tokens", 0)),
                    "output_tokens": int(usage.get("output_tokens", 0)),
                    "parse_error": output.get("parse_error"),
                }
            )
            if progress is not None:
                progress.update(1)

    if progress is not None:
        progress.close()

    by_task = {}
    for task in task_names:
        task_samples = [sample for sample in samples if sample["task"] == task]
        by_task[task] = speed_metrics(task_samples)
    return samples, {"samples_per_task": len(rows), "warmup_per_task": warmup, "by_task": by_task}


def _speed_progress(total: int):
    try:
        from tqdm.auto import tqdm
    except ImportError as exc:
        raise RuntimeError("Speed progress requires tqdm. Install it with: pip install tqdm") from exc
    return tqdm(total=total, desc="speed", unit="request", dynamic_ncols=True)


def speed_metrics(samples: list[dict]) -> dict:
    model_latencies = [float(row["model_latency_ms"]) for row in samples]
    wall_latencies = [float(row["wall_latency_ms"]) for row in samples]
    total_model_seconds = sum(model_latencies) / 1000
    total_wall_seconds = sum(wall_latencies) / 1000
    input_tokens = sum(int(row["input_tokens"]) for row in samples)
    output_tokens = sum(int(row["output_tokens"]) for row in samples)
    return {
        "requests": len(samples),
        "model_latency_ms": _latency_summary(model_latencies),
        "wall_latency_ms": _latency_summary(wall_latencies),
        "requests_per_second": len(samples) / total_wall_seconds if total_wall_seconds else 0.0,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_tokens_per_second": input_tokens / total_model_seconds if total_model_seconds else 0.0,
        "output_tokens_per_second": output_tokens / total_model_seconds if total_model_seconds else 0.0,
        "parse_success_rate": _success_rate(row.get("parse_error") for row in samples),
    }


def truncate_body_text(rows: list[dict], max_body_chars: int | None) -> list[dict]:
    if max_body_chars is None:
        return list(rows)
    if max_body_chars < 1:
        raise ValueError("max_body_chars must be positive")
    truncated = []
    for row in rows:
        item = dict(row)
        item["body_text"] = str(row.get("body_text", ""))[:max_body_chars]
        truncated.append(item)
    return truncated


def select_benchmark_rows(
    rows: list[dict],
    limit: int | None,
    *,
    seed: int = 20260629,
    label_mode: str = "binary",
) -> list[dict]:
    if limit is None or limit >= len(rows):
        return list(rows)
    if limit < 1:
        return []
    if label_mode not in {"binary", "multiclass"}:
        raise ValueError(f"unsupported label mode: {label_mode}")

    groups: dict[str, list[dict]] = {}
    for row in rows:
        label = _gold_category(row) if label_mode == "multiclass" else _gold_spam_label(row)
        if label:
            groups.setdefault(label, []).append(row)
    allocations = _balanced_allocations({label: len(items) for label, items in groups.items()}, limit)
    selected = []
    for label, allocation in allocations.items():
        ordered = sorted(
            groups[label],
            key=lambda row: (len(str(row.get("body_text", ""))), _stable_key(seed, str(row.get("email_id", "")))),
        )
        selected.extend(_evenly_spaced(ordered, allocation))
    return sorted(selected, key=lambda row: _stable_key(seed, str(row.get("email_id", ""))))


def _balanced_allocations(sizes: dict[str, int], limit: int) -> dict[str, int]:
    allocations = {label: 0 for label in sorted(sizes)}
    while sum(allocations.values()) < limit:
        eligible = [label for label in allocations if allocations[label] < sizes[label]]
        if not eligible:
            break
        for label in eligible:
            if sum(allocations.values()) >= limit:
                break
            allocations[label] += 1
    return allocations

def _task_runners() -> dict[str, Callable[[Email, LLMClient], dict]]:
    classifier = ClassifyEmailSkill()
    summarizer = SummarizeEmailSkill()
    actions = ExtractActionItemsSkill()
    reply = DraftReplySkill()
    return {
        "classify_email": lambda email, llm: classifier.run(email, {}, llm),
        "summarize_email": lambda email, llm: summarizer.run(email, {}, llm),
        "extract_action_items": lambda email, llm: actions.run(email, {}, llm),
        "draft_reply": lambda email, llm: reply.run(email, {}, llm),
    }


def _gold_category(row: dict) -> str | None:
    labels = row.get("labels") or {}
    value = labels.get("category", row.get("category_label"))
    return str(value) if value in VALID_CATEGORIES else None


def _wrong_category(gold: str) -> str:
    return next(category for category in sorted(VALID_CATEGORIES) if category != gold)


def _gold_spam_label(row: dict) -> str:
    labels = row.get("labels") or {}
    value = labels.get("spam_label")
    if value in {"spam", "ham"}:
        return value
    if "spam" in labels:
        return "spam" if bool(labels["spam"]) else "ham"
    return "spam" if labels.get("category") == "spam" else "ham"


def _multiclass_confusion(y_true: list[str], y_pred: list[str]) -> dict[str, dict[str, int]]:
    labels = sorted(set(y_true) | set(y_pred))
    return {
        true_label: {
            predicted_label: sum(t == true_label and p == predicted_label for t, p in zip(y_true, y_pred))
            for predicted_label in labels
        }
        for true_label in labels
    }


def _binary_confusion(y_true: list[str], y_pred: list[str]) -> dict[str, int]:
    return {
        "tp": sum(t == "spam" and p == "spam" for t, p in zip(y_true, y_pred)),
        "fp": sum(t == "ham" and p == "spam" for t, p in zip(y_true, y_pred)),
        "fn": sum(t == "spam" and p == "ham" for t, p in zip(y_true, y_pred)),
        "tn": sum(t == "ham" and p == "ham" for t, p in zip(y_true, y_pred)),
    }


def _latency_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
    return {
        "mean": sum(values) / len(values),
        "p50": _percentile(values, 50),
        "p95": _percentile(values, 95),
        "p99": _percentile(values, 99),
    }


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _success_rate(errors: Iterable[Any]) -> float:
    values = list(errors)
    if not values:
        return 1.0
    return sum(not error for error in values) / len(values)


def _evenly_spaced(rows: list[dict], count: int) -> list[dict]:
    if count <= 0 or not rows:
        return []
    if count >= len(rows):
        return rows
    if count == 1:
        return [rows[len(rows) // 2]]
    return [rows[round(index * (len(rows) - 1) / (count - 1))] for index in range(count)]


def _stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()
