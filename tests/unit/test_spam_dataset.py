import tempfile
import unittest
from pathlib import Path
from email.message import EmailMessage

from email_system.evaluation.spam_dataset import _message_body, content_fingerprint, iter_enron_rows, iter_maildir_rows, split_records, write_dataset


def record(email_id: str, label: str, body: str) -> dict:
    return {
        "email_id": email_id,
        "subject": "subject",
        "body_text": body,
        "labels": {"spam_label": label},
        "source": "fixture",
    }


class SpamDatasetTest(unittest.TestCase):
    def test_split_is_deterministic_and_stratified(self):
        rows = [record(f"spam-{i}", "spam", f"spam body {i}") for i in range(10)]
        rows += [record(f"ham-{i}", "ham", f"ham body {i}") for i in range(10)]

        first, first_manifest = split_records(rows, seed=7)
        second, second_manifest = split_records(reversed(rows), seed=7)

        self.assertEqual(first, second)
        self.assertEqual(first_manifest, second_manifest)
        self.assertEqual(first_manifest["splits"]["train"]["labels"], {"ham": 7, "spam": 7})
        self.assertEqual(first_manifest["splits"]["validation"]["labels"], {"ham": 1, "spam": 1})
        self.assertEqual(first_manifest["splits"]["test"]["labels"], {"ham": 2, "spam": 2})

    def test_duplicate_content_is_removed_before_split(self):
        rows = [
            record("one", "spam", "same body"),
            record("two", "spam", "same   body"),
            record("three", "ham", "different body"),
        ]

        splits, manifest = split_records(rows)
        fingerprints = [content_fingerprint(item) for values in splits.values() for item in values]

        self.assertEqual(len(fingerprints), len(set(fingerprints)))
        self.assertEqual(manifest["duplicates_removed"], 1)
        self.assertEqual(manifest["unique_records"], 2)

    def test_enron_reader_tolerates_nul_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "enron.csv"
            path.write_bytes(
                b"Message ID,Subject,Message,Spam/Ham,Date\n"
                b"1,Hello,normal\x00 body,ham,2026-01-01\n"
            )

            rows = list(iter_enron_rows(path))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["labels"]["spam_label"], "ham")
        self.assertEqual(rows[0]["body_text"], "normal body")

    def test_maildir_reader_maps_message_and_folder_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "user-a" / "inbox" / "1."
            path.parent.mkdir(parents=True)
            message = EmailMessage()
            message["Message-ID"] = "<mail-1@example.com>"
            message["From"] = "Alice <alice@example.com>"
            message["To"] = "Bob <bob@example.com>"
            message["Subject"] = "Project update"
            message.set_content("The project is on schedule.")
            path.write_bytes(message.as_bytes())

            rows = list(iter_maildir_rows(directory))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["from"], "alice@example.com")
        self.assertEqual(rows[0]["to"], ["bob@example.com"])
        self.assertEqual(rows[0]["labels"]["spam_label"], "ham")
        self.assertNotIn("category", rows[0]["labels"])
        self.assertEqual(rows[0]["metadata"]["maildir_folder"], "inbox")


    def test_maildir_reader_tolerates_malformed_address_header(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "user-a" / "inbox" / "broken."
            path.parent.mkdir(parents=True)
            path.write_bytes(
                b"Message-ID: <broken@example.com>\n"
                b"From: sender@example.com\n"
                b"To: Broken Group @enron.com, .name@enron.com\n"
                b"Subject: malformed recipients\n\nbody\n"
            )

            rows = list(iter_maildir_rows(directory))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["subject"], "malformed recipients")
        self.assertEqual(rows[0]["body_text"], "body")


    def test_malformed_multipart_body_does_not_abort_dataset(self):
        message = EmailMessage()
        message.set_type("multipart/mixed")
        message.set_payload(["broken part"])

        self.assertEqual(_message_body(message), "")

    def test_unknown_charset_falls_back_to_utf8(self):
        message = EmailMessage()
        message.set_payload(b"hello")
        message.set_param("charset", "charset=")

        self.assertEqual(_message_body(message), "hello")

    def test_write_dataset_creates_manifest_and_splits(self):
        rows = [record(f"row-{i}", "spam" if i % 2 else "ham", f"body {i}") for i in range(10)]
        splits, manifest = split_records(rows)
        with tempfile.TemporaryDirectory() as directory:
            write_dataset(directory, splits, manifest)
            output = Path(directory)
            self.assertTrue((output / "train.jsonl").exists())
            self.assertTrue((output / "validation.jsonl").exists())
            self.assertTrue((output / "test.jsonl").exists())
            self.assertTrue((output / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
