import json
import tempfile
import unittest
from pathlib import Path

from email_system.evaluation.phishing_dataset import iter_phishing_rows, split_records, write_dataset


def record(email_id: str, label: str, body: str) -> dict:
    is_phishing = label == "phishing"
    return {
        "email_id": email_id,
        "subject": "subject",
        "body_text": body,
        "labels": {"phishing_label": label, "spam_label": "spam" if is_phishing else "ham"},
        "source": "fixture",
    }


class PhishingDatasetTest(unittest.TestCase):
    def test_reader_maps_labels_to_email_system_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "CEAS_08.csv").write_text(
                "sender,receiver,date,subject,body,label,urls\n"
                "sender@example.com,me@example.com,2026-01-01,Hello,Click here,1,2\n"
                "friend@example.com,me@example.com,2026-01-02,Hi,Normal message,0,0\n",
                encoding="utf-8",
            )

            rows = list(iter_phishing_rows(root))

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["labels"]["phishing_label"], "phishing")
        self.assertEqual(rows[0]["labels"]["category"], "spam")
        self.assertTrue(rows[0]["labels"]["phishing"])
        self.assertEqual(rows[0]["metadata"], {"urls": 2})
        self.assertEqual(rows[1]["labels"]["phishing_label"], "legitimate")
        self.assertEqual(rows[1]["labels"]["category"], "other")

    def test_text_combined_file_uses_body_text(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "phishing_email.csv").write_text("text_combined,label\ncombined text,1\n", encoding="utf-8")

            rows = list(iter_phishing_rows(root))

        self.assertEqual(rows[0]["subject"], "")
        self.assertEqual(rows[0]["body_text"], "combined text")

    def test_split_is_deterministic_and_deduplicated(self):
        rows = [record(f"p-{i}", "phishing", f"phishing body {i}") for i in range(10)]
        rows += [record(f"l-{i}", "legitimate", f"legit body {i}") for i in range(10)]
        rows.append(record("duplicate", "phishing", "phishing   body 1"))

        first, first_manifest = split_records(rows, seed=11)
        second, second_manifest = split_records(reversed(rows), seed=11)

        self.assertEqual(first, second)
        self.assertEqual(first_manifest, second_manifest)
        self.assertEqual(first_manifest["duplicates_removed"], 1)
        self.assertEqual(first_manifest["splits"]["train"]["labels"], {"legitimate": 7, "phishing": 7})

    def test_write_dataset_creates_manifest_and_splits(self):
        rows = [record(f"row-{i}", "phishing" if i % 2 else "legitimate", f"body {i}") for i in range(10)]
        splits, manifest = split_records(rows)
        with tempfile.TemporaryDirectory() as directory:
            write_dataset(directory, splits, manifest)
            output = Path(directory)
            self.assertTrue((output / "train.jsonl").exists())
            self.assertTrue((output / "validation.jsonl").exists())
            self.assertTrue((output / "test.jsonl").exists())
            loaded = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(loaded["task"], "phishing_detection")


if __name__ == "__main__":
    unittest.main()
