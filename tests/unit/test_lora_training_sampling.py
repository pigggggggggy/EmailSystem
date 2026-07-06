import argparse
import unittest

from training.train_lora_classification import (
    _configure_logical_epoch_strategies,
    _drop_reserved_labels_column,
    _limit_dataset,
    _prepare_train_dataset,
)


class FakeSplit:
    def __init__(self, values):
        self.values = list(values)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index]

    @property
    def column_names(self):
        if not self.values or not isinstance(self.values[0], dict):
            return []
        return list(self.values[0])

    def remove_columns(self, names):
        return FakeSplit(
            [{key: value for key, value in item.items() if key not in names} for item in self.values]
        )

    def shuffle(self, *, seed):
        offset = seed % len(self.values)
        return FakeSplit(self.values[offset:] + self.values[:offset])

    def select(self, indexes):
        return FakeSplit([self.values[index] for index in indexes])


def concat(splits):
    values = []
    for split in splits:
        values.extend(split.values)
    return FakeSplit(values)


def args(**overrides):
    values = {
        "epochs": 3.0,
        "max_train_samples": 4,
        "seed": 11,
        "resample_train_each_epoch": True,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 2,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "eval_steps": 200,
        "save_steps": 200,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class LoraTrainingSamplingTest(unittest.TestCase):
    def test_resamples_and_concatenates_one_subset_per_epoch(self):
        parsed = args()
        sampled = _prepare_train_dataset(FakeSplit(list(range(10))), parsed, concat)

        self.assertEqual(len(sampled), 12)
        self.assertEqual(sampled.values[:4], [1, 2, 3, 4])
        self.assertEqual(sampled.values[4:8], [2, 3, 4, 5])
        self.assertEqual(sampled.values[8:12], [3, 4, 5, 6])
        self.assertEqual(parsed._logical_epochs, 3)
        self.assertEqual(parsed._effective_epochs, 1.0)
        self.assertEqual(parsed._steps_per_logical_epoch, 1)

    def test_epoch_strategies_map_to_logical_epoch_steps(self):
        parsed = args(max_train_samples=9, per_device_train_batch_size=2, gradient_accumulation_steps=2)
        _prepare_train_dataset(FakeSplit(list(range(20))), parsed, concat)
        _configure_logical_epoch_strategies(parsed)

        self.assertEqual(parsed._effective_eval_strategy, "steps")
        self.assertEqual(parsed._effective_save_strategy, "steps")
        self.assertEqual(parsed._effective_eval_steps, 3)
        self.assertEqual(parsed._effective_save_steps, 3)

    def test_can_disable_resampling(self):
        parsed = args(resample_train_each_epoch=False)
        sampled = _prepare_train_dataset(FakeSplit(list(range(10))), parsed, concat)

        self.assertEqual(len(sampled), 4)
        self.assertEqual(parsed._effective_epochs, 3.0)
        self.assertIsNone(parsed._steps_per_logical_epoch)

    def test_soft_balances_multiclass_categories_with_oversampling(self):
        values = [
            {"category_label": "spam", "id": index} for index in range(100)
        ] + [
            {"category_label": "support", "id": 100 + index} for index in range(25)
        ] + [
            {"category_label": "invoice", "id": 125},
        ]
        parsed = args(
            epochs=1.0,
            max_train_samples=60,
            balance_category_labels=True,
            category_balance_max_ratio=3.0,
        )

        sampled = _prepare_train_dataset(FakeSplit(values), parsed, concat)

        counts = {}
        for item in sampled.values:
            counts[item["category_label"]] = counts.get(item["category_label"], 0) + 1
        self.assertEqual(sum(counts.values()), 60)
        self.assertGreater(counts["spam"], counts["support"])
        self.assertGreater(counts["support"], counts["invoice"])
        self.assertLessEqual(counts["spam"] / counts["invoice"], 3.0)

    def test_validation_balancing_does_not_duplicate_rows(self):
        values = [
            {"category_label": "spam", "id": index} for index in range(20)
        ] + [
            {"category_label": "invoice", "id": 20 + index} for index in range(2)
        ]

        sampled = _limit_dataset(
            FakeSplit(values),
            20,
            seed=7,
            balance_category_labels=True,
            category_balance_max_ratio=3.0,
            allow_oversampling=False,
        )

        ids = [item["id"] for item in sampled.values]
        counts = {}
        for item in sampled.values:
            counts[item["category_label"]] = counts.get(item["category_label"], 0) + 1
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(counts, {"invoice": 2, "spam": 6})

    def test_drops_reserved_labels_but_keeps_multiclass_target(self):
        split = FakeSplit(
            [
                {
                    "messages": [{"role": "assistant", "content": "target"}],
                    "category_label": "support",
                    "labels": {"category": "support", "spam_label": "ham"},
                }
            ]
        )

        cleaned = _drop_reserved_labels_column(split)

        self.assertNotIn("labels", cleaned.values[0])
        self.assertEqual(cleaned.values[0]["category_label"], "support")
        self.assertIn("messages", cleaned.values[0])


if __name__ == "__main__":
    unittest.main()
