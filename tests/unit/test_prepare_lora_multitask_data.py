import json
import unittest

from training.prepare_lora_multitask_data import (
    build_multitask_item,
    _tasks_for_row,
)


class PrepareLoraMultitaskDataTest(unittest.TestCase):
    def setUp(self):
        self.row = {
            "email_id": "mail-1",
            "source": "fixture",
            "subject": "Please review the contract",
            "from": "sender@example.com",
            "to": ["me@example.com"],
            "timestamp": "2026-07-07",
            "body_text": "Please review the contract and confirm by Friday.",
            "labels": {"category": "business_email", "spam_label": "ham"},
            "category_label": "business_email",
            "consensus_confidence": 0.91,
        }

    def test_classification_item_uses_multiclass_target(self):
        item = build_multitask_item(self.row, "classify_email", max_body_chars=200)

        self.assertEqual(item["task"], "classify_email")
        self.assertEqual(item["category_label"], "business_email")
        target = json.loads(item["messages"][-1]["content"])
        self.assertEqual(target["category"], "business_email")
        self.assertEqual(target["priority"], "normal")
        self.assertEqual(target["confidence"], 0.91)

    def test_generates_runtime_compatible_task_targets(self):
        summary = build_multitask_item(self.row, "summarize_email", max_body_chars=200)
        actions = build_multitask_item(self.row, "extract_action_items", max_body_chars=200)
        reply = build_multitask_item(self.row, "draft_reply", max_body_chars=200)

        self.assertIn("summary", json.loads(summary["messages"][-1]["content"]))
        self.assertTrue(json.loads(actions["messages"][-1]["content"])["action_items"])
        self.assertIn("邮件", reply["messages"][-1]["content"])

    def test_spam_reply_target_is_cautious(self):
        row = dict(self.row)
        row["labels"] = {"category": "spam", "spam_label": "spam"}
        item = build_multitask_item(row, "draft_reply", max_body_chars=200)

        self.assertIn("不要点击链接", item["messages"][-1]["content"])

    def test_default_style_task_sampling_keeps_classification_for_all_rows(self):
        tasks = _tasks_for_row(
            self.row,
            {"classify_email": 10, "summarize_email": 0, "extract_action_items": 0, "draft_reply": 0},
            seed=1,
        )

        self.assertEqual(tasks, ["classify_email"])


if __name__ == "__main__":
    unittest.main()
