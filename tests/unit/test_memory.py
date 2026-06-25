import tempfile
import unittest
from pathlib import Path

from email_system.memory import JsonlLongTermMemory, MemoryRecord, ShortTermMemory
from email_system.schemas import Email


def email(email_id="e-1", thread_id="t-1", sender="a@example.com"):
    return Email.from_dict(
        {
            "email_id": email_id,
            "thread_id": thread_id,
            "subject": "Login issue",
            "from": sender,
            "to": ["support@example.com"],
            "body_text": "Please check my login issue.",
        }
    )


class MemoryTest(unittest.TestCase):
    def test_short_term_memory_keeps_thread_context(self):
        memory = ShortTermMemory(max_items_per_thread=2)
        memory.save(email("e-1"), {"summary": "first", "category": "support"})
        memory.save(email("e-2"), {"summary": "second", "category": "support"})

        context = memory.load(email("e-3"))

        self.assertEqual(context["email_ids"], ["e-1", "e-2"])
        self.assertEqual(context["recent_summaries"], ["first", "second"])

    def test_jsonl_long_term_memory_searches_related_records(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonlLongTermMemory(Path(tmpdir) / "memory.jsonl")
            store.save(
                MemoryRecord(
                    email_id="e-1",
                    thread_id="t-1",
                    subject="Login issue",
                    sender="a@example.com",
                    category="support",
                    priority="normal",
                    summary="first",
                    action_count=1,
                )
            )

            matches = store.search(email("e-2", thread_id="t-1"))

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].email_id, "e-1")


if __name__ == "__main__":
    unittest.main()
