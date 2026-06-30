import json
import tempfile
import unittest
from pathlib import Path
from typing import List

from training.prepare_lora_classification_data import _collect_split_rows, _write_chat_jsonl


def row(email_id: str, label: str) -> dict:
    return {
        "email_id": email_id,
        "subject": f"subject {email_id}",
        "from": "sender@example.com",
        "to": ["me@example.com"],
        "timestamp": "2026-01-01",
        "body_text": f"body {email_id}",
        "labels": {"spam_label": label},
        "source": "fixture",
    }


class PrepareLoraDataTest(unittest.TestCase):
    def test_collect_split_rows_merges_input_dirs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()
            _write_jsonl(first / "train.jsonl", [row("a", "spam")])
            _write_jsonl(second / "train.jsonl", [row("b", "ham"), row("c", "spam")])

            rows, sources = _collect_split_rows([first, second], "train")

        self.assertEqual([item["email_id"] for item in rows], ["a", "b", "c"])
        self.assertEqual(dict(sources), {str(first): 1, str(second): 2})

    def test_write_chat_jsonl_preserves_spam_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "train.jsonl"
            count = _write_chat_jsonl(
                output,
                [row("spam-1", "spam"), row("ham-1", "ham")],
                max_body_chars=100,
                spam_confidence=0.95,
                ham_confidence=0.90,
            )
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(count, 2)
        self.assertEqual(rows[0]["messages"][-1]["content"], '{"category":"spam","priority":"normal","confidence":0.95}')
        self.assertEqual(rows[1]["messages"][-1]["content"], '{"category":"other","priority":"normal","confidence":0.9}')


def _write_jsonl(path: Path, rows: List[dict]) -> None:
    path.write_text("".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
