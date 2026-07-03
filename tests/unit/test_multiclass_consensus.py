import json
import tempfile
import unittest
from pathlib import Path

from training.label_multiclass_consensus import (
    build_request_payload,
    build_training_item,
    consensus_decision,
    extract_content,
    prepare_config,
    stable_sample,
    validate_label,
)


MODELS = ("model-a", "model-b")


class MulticlassConsensusTest(unittest.TestCase):
    def test_accepts_only_matching_categories(self):
        labels = {
            "model-a": {"category": "support", "confidence": 0.9},
            "model-b": {"category": "support", "confidence": 0.8},
        }
        decision = consensus_decision(labels, MODELS)
        self.assertEqual(decision, {"status": "accepted", "category": "support", "confidence": 0.8})

    def test_records_disagreement_and_errors(self):
        disagreement = consensus_decision(
            {
                "model-a": {"category": "sales", "confidence": 0.9},
                "model-b": {"category": "spam", "confidence": 0.9},
            },
            MODELS,
        )
        error = consensus_decision({"model-a": {"error": "timeout"}}, MODELS)
        self.assertEqual(disagreement["status"], "disagreement")
        self.assertEqual(error["status"], "error")

    def test_validates_openai_response_label(self):
        content = extract_content({"choices": [{"message": {"content": '{"category":"invoice","confidence":0.7}'}}]})
        self.assertEqual(validate_label(json.loads(content)), {"category": "invoice", "confidence": 0.7})
        with self.assertRaises(ValueError):
            validate_label({"category": "unknown", "confidence": 1.0})


    def test_requests_structured_output_without_thinking(self):
        payload = build_request_payload("model-a", "email")
        self.assertEqual(payload["max_tokens"], 256)
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})

    def test_falls_back_to_reasoning_content_when_content_is_empty(self):
        content = extract_content(
            {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "reasoning_content": '{"category":"spam","confidence":0.9}',
                        }
                    }
                ]
            }
        )
        self.assertEqual(json.loads(content)["category"], "spam")

    def test_builds_runtime_compatible_multiclass_training_item(self):
        row = {
            "email_id": "mail-1",
            "subject": "Need help",
            "body_text": "The service is broken",
            "labels": {"spam_label": "ham"},
        }
        annotation = {
            "consensus_category": "support",
            "consensus_confidence": 0.85,
            "model_labels": {"model-a": {}, "model-b": {}},
        }
        item = build_training_item(row, annotation, max_body_chars=100)
        self.assertEqual(item["labels"]["category"], "support")
        self.assertEqual(item["category_label"], "support")
        self.assertEqual(json.loads(item["messages"][-1]["content"])["category"], "support")
        self.assertEqual(item["body_text"], "The service is broken")

    def test_sampling_and_resume_configuration_are_deterministic(self):
        rows = [{"email_id": f"mail-{index}"} for index in range(10)]
        self.assertEqual(stable_sample(rows, limit=4, seed=3), stable_sample(reversed(rows), limit=4, seed=3))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            prepare_config(path, {"models": list(MODELS)})
            prepare_config(path, {"models": list(MODELS)})
            with self.assertRaises(SystemExit):
                prepare_config(path, {"models": ["different", "models"]})


if __name__ == "__main__":
    unittest.main()
