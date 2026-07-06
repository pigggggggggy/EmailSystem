import json
import unittest

from email_system.evaluation.independent_benchmark import (
    run_classification_quality,
    run_task_speed,
    select_benchmark_rows,
    speed_metrics,
    truncate_body_text,
)
from email_system.models import GenerationResult


class InvalidLLM:
    def generate(self, prompt, *, task, max_tokens=512):
        return GenerationResult(text="not json", input_tokens=10, output_tokens=2, latency_ms=5.0)


class LowConfidenceLLM:
    def generate(self, prompt, *, task, max_tokens=512):
        text = json.dumps({"category": "spam", "priority": "normal", "confidence": 0.0})
        return GenerationResult(text=text, input_tokens=10, output_tokens=5, latency_ms=5.0)


class FixtureLLM:
    def generate(self, prompt, *, task, max_tokens=512):
        if task == "classify_email":
            category = "spam" if "BUY NOW" in prompt else "personal_email"
            text = json.dumps({"category": category, "priority": "normal", "confidence": 0.9})
        elif task == "summarize_email":
            text = json.dumps({"summary": "summary", "confidence": 0.8})
        elif task == "extract_action_items":
            text = json.dumps({"action_items": []})
        else:
            text = "reply"
        return GenerationResult(text=text, input_tokens=20, output_tokens=5, latency_ms=10.0)


def row(index, label, body):
    return {
        "email_id": f"email-{index}",
        "subject": "subject",
        "from": "sender@example.com",
        "body_text": body,
        "labels": {"spam_label": label, "spam": label == "spam"},
    }


class IndependentBenchmarkTest(unittest.TestCase):
    def test_quality_only_scores_binary_classification(self):
        rows = [row(1, "spam", "BUY NOW"), row(2, "ham", "Hello friend")]

        predictions, metrics = run_classification_quality(FixtureLLM(), rows)

        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["macro_f1"], 1.0)
        self.assertEqual(metrics["spam_recall"], 1.0)
        self.assertEqual(metrics["parse_success_rate"], 1.0)
        self.assertEqual(len(predictions), 2)

    def test_low_confidence_is_not_auto_accepted(self):
        predictions, metrics = run_classification_quality(LowConfidenceLLM(), [row(1, "spam", "BUY NOW")])

        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["low_confidence_rate"], 1.0)
        self.assertEqual(metrics["accepted_coverage"], 0.0)
        self.assertFalse(predictions[0]["accepted_prediction"])

    def test_parse_failure_cannot_accidentally_score_ham_as_correct(self):
        _, metrics = run_classification_quality(InvalidLLM(), [row(1, "ham", "hello")])

        self.assertEqual(metrics["accuracy"], 0.0)
        self.assertEqual(metrics["parse_success_rate"], 0.0)
        self.assertEqual(metrics["valid_category_rate"], 0.0)

    def test_speed_is_reported_separately_for_each_task(self):
        rows = [row(1, "spam", "BUY NOW"), row(2, "ham", "Hello friend")]

        samples, metrics = run_task_speed(FixtureLLM(), rows, warmup=0)

        self.assertEqual(len(samples), 8)
        self.assertEqual(set(metrics["by_task"]), {"classify_email", "summarize_email", "extract_action_items", "draft_reply"})
        self.assertEqual(metrics["by_task"]["classify_email"]["output_tokens"], 10)

    def test_speed_percentiles_and_token_rates(self):
        metrics = speed_metrics(
            [
                {"model_latency_ms": 10, "wall_latency_ms": 20, "input_tokens": 10, "output_tokens": 2, "parse_error": None},
                {"model_latency_ms": 30, "wall_latency_ms": 40, "input_tokens": 20, "output_tokens": 4, "parse_error": "bad"},
            ]
        )

        self.assertEqual(metrics["model_latency_ms"]["p50"], 20)
        self.assertEqual(metrics["wall_latency_ms"]["p95"], 39)
        self.assertEqual(metrics["parse_success_rate"], 0.5)
        self.assertAlmostEqual(metrics["output_tokens_per_second"], 150.0)

    def test_body_truncation_is_fixed_and_non_mutating(self):
        rows = [row(1, "ham", "abcdefgh")]

        truncated = truncate_body_text(rows, 4)

        self.assertEqual(truncated[0]["body_text"], "abcd")
        self.assertEqual(rows[0]["body_text"], "abcdefgh")

    def test_speed_selection_is_balanced_and_deterministic(self):
        rows = [row(i, "spam" if i < 10 else "ham", "x" * (i + 1)) for i in range(20)]

        first = select_benchmark_rows(rows, 6, seed=9)
        second = select_benchmark_rows(list(reversed(rows)), 6, seed=9)

        self.assertEqual(first, second)
        labels = [item["labels"]["spam_label"] for item in first]
        self.assertEqual(labels.count("spam"), 3)
        self.assertEqual(labels.count("ham"), 3)

    def test_multiclass_quality_uses_category_labels(self):
        rows = [
            {**row(1, "ham", "hello"), "labels": {"category": "personal_email"}},
            {**row(2, "spam", "BUY NOW"), "labels": {"category": "spam"}},
        ]

        predictions, metrics = run_classification_quality(FixtureLLM(), rows)

        self.assertEqual(metrics["quality_mode"], "multiclass")
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(predictions[0]["gold_category"], "personal_email")
        self.assertIn("personal_email", metrics["confusion_matrix"])

    def test_multiclass_selection_balances_available_categories(self):
        rows = []
        for category in ("automated_email", "business_email", "spam"):
            for index in range(5):
                item = row(len(rows), "spam" if category == "spam" else "ham", category)
                item["labels"]["category"] = category
                rows.append(item)

        selected = select_benchmark_rows(rows, 6, seed=5, label_mode="multiclass")

        counts = {}
        for item in selected:
            category = item["labels"]["category"]
            counts[category] = counts.get(category, 0) + 1
        self.assertEqual(counts, {"automated_email": 2, "spam": 2, "business_email": 2})


if __name__ == "__main__":
    unittest.main()
