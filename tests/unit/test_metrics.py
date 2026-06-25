import unittest

from email_system.evaluation import classification_metrics


class MetricsTest(unittest.TestCase):
    def test_classification_metrics_macro_f1(self):
        metrics = classification_metrics(
            ["support", "invoice", "support", "sales"],
            ["support", "invoice", "sales", "sales"],
        )

        self.assertEqual(metrics["accuracy"], 0.75)
        self.assertEqual(round(metrics["macro_f1"], 3), 0.778)
        self.assertEqual(metrics["per_class"]["support"]["support"], 2)
        self.assertEqual(round(metrics["per_class"]["support"]["recall"], 3), 0.5)


if __name__ == "__main__":
    unittest.main()
