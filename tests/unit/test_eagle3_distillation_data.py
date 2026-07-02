import json
import tempfile
import unittest
from pathlib import Path

from training.generate_eagle3_distillation_data import (
    TASKS,
    build_candidates,
    prepare_generation_config,
    stable_sample,
    validate_teacher_output,
)


class Eagle3DistillationDataTest(unittest.TestCase):
    def test_builds_one_conversation_candidate_per_task(self):
        rows = [{"email_id": "mail-1", "subject": "Hello", "body_text": "Body"}]
        candidates = build_candidates(rows, max_body_chars=100)
        self.assertEqual(len(candidates), len(TASKS))
        self.assertEqual({item["task"] for item in candidates}, set(TASKS))
        self.assertTrue(all(item["id"].startswith("mail-1:") for item in candidates))

    def test_sampling_is_deterministic(self):
        rows = [{"email_id": f"mail-{index}"} for index in range(10)]
        first = stable_sample(rows, limit=4, seed=7)
        second = stable_sample(reversed(rows), limit=4, seed=7)
        self.assertEqual(first, second)

    def test_validates_and_canonicalizes_json_tasks(self):
        classification = validate_teacher_output(
            "classify_email", '{"priority":"normal","confidence":0.8,"category":"spam"}'
        )
        summary = validate_teacher_output("summarize_email", '{"summary":"hello","confidence":0.8}')
        actions = validate_teacher_output("extract_action_items", '{"action_items":[]}')
        self.assertEqual(json.loads(classification)["category"], "spam")
        self.assertEqual(json.loads(summary)["summary"], "hello")
        self.assertEqual(json.loads(actions)["action_items"], [])

    def test_rejects_invalid_or_truncated_teacher_outputs(self):
        with self.assertRaises(ValueError):
            validate_teacher_output("classify_email", '{"category":"7"}')
        with self.assertRaises(ValueError):
            validate_teacher_output("draft_reply", "short")
        with self.assertRaises(ValueError):
            validate_teacher_output("draft_reply", "long enough reply", finish_reason="length")

    def test_resume_configuration_must_match(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "generation_config.json"
            prepare_generation_config(path, {"limit": 10})
            prepare_generation_config(path, {"limit": 10})
            with self.assertRaises(SystemExit):
                prepare_generation_config(path, {"limit": 20})


if __name__ == "__main__":
    unittest.main()
