from __future__ import annotations

from dataclasses import dataclass

from .metrics import classification_metrics, latency_metrics


@dataclass(frozen=True)
class EvaluationResult:
    metrics: dict
    predictions: list[dict]


def evaluate_predictions(gold_rows: list[dict], predictions: list[dict]) -> EvaluationResult:
    gold_by_id = {row["email_id"]: row for row in gold_rows}
    y_true = []
    y_pred = []
    for prediction in predictions:
        gold = gold_by_id.get(prediction["email_id"])
        if not gold:
            continue
        category = (gold.get("labels") or {}).get("category")
        if category is None:
            continue
        y_true.append(category)
        y_pred.append(prediction.get("category", "other"))

    total_skill_outputs = sum(len(prediction.get("timings_ms", {})) for prediction in predictions)
    total_skill_errors = sum(len(prediction.get("skill_errors", {})) for prediction in predictions)
    parse_success_rate = (
        (total_skill_outputs - total_skill_errors) / total_skill_outputs if total_skill_outputs else 1.0
    )

    metrics = {
        "classification": classification_metrics(y_true, y_pred),
        "latency": latency_metrics(predictions),
        "parse_success_rate": parse_success_rate,
        "parse_errors": total_skill_errors,
    }
    return EvaluationResult(metrics=metrics, predictions=predictions)
