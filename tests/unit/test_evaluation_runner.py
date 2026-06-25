import unittest

from email_system.evaluation import evaluate_predictions


class EvaluationRunnerTest(unittest.TestCase):
    def test_parse_success_rate_uses_skill_errors(self):
        result = evaluate_predictions(
            [{"email_id": "e-1", "labels": {"category": "other"}}],
            [
                {
                    "email_id": "e-1",
                    "category": "other",
                    "timings_ms": {"classify_email": 1.0, "summarize_email": 1.0},
                    "skill_errors": {"classify_email": "No JSON object found"},
                }
            ],
        )

        self.assertEqual(result.metrics["parse_errors"], 1)
        self.assertEqual(result.metrics["parse_success_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
