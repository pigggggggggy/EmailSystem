from __future__ import annotations

from collections import Counter
from statistics import mean


def classification_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not y_true:
        return {"accuracy": 0.0, "macro_f1": 0.0, "per_class": {}}

    labels = sorted(set(y_true) | set(y_pred))
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    per_class = {}
    f1_scores = []
    for label in labels:
        tp = sum(1 for true, pred in zip(y_true, y_pred) if true == label and pred == label)
        fp = sum(1 for true, pred in zip(y_true, y_pred) if true != label and pred == label)
        fn = sum(1 for true, pred in zip(y_true, y_pred) if true == label and pred != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_scores.append(f1)
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": Counter(y_true)[label]}

    return {"accuracy": correct / len(y_true), "macro_f1": mean(f1_scores), "per_class": per_class}


def latency_metrics(predictions: list[dict]) -> dict:
    totals = []
    by_skill: dict[str, list[float]] = {}
    for row in predictions:
        timings = row.get("timings_ms", {})
        totals.append(sum(timings.values()))
        for name, value in timings.items():
            by_skill.setdefault(name, []).append(float(value))

    return {
        "emails": len(predictions),
        "avg_end_to_end_ms": mean(totals) if totals else 0.0,
        "avg_skill_ms": {name: mean(values) for name, values in sorted(by_skill.items())},
    }
