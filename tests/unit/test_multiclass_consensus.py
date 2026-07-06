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
    _weighted_allocations,
    validate_input_weights,
)


MODELS = ("model-a", "model-b")


class MulticlassConsensusTest(unittest.TestCase):
    def test_accepts_only_matching_categories(self):
        labels = {
            "model-a": {"category": "business_email", "confidence": 0.9},
            "model-b": {"category": "business_email", "confidence": 0.8},
        }
        decision = consensus_decision(labels, MODELS)
        self.assertEqual(decision, {"status": "accepted", "category": "business_email", "confidence": 0.8})

    def test_records_disagreement_and_errors(self):
        disagreement = consensus_decision(
            {
                "model-a": {"category": "marketing_email", "confidence": 0.9},
                "model-b": {"category": "spam", "confidence": 0.9},
            },
            MODELS,
        )
        error = consensus_decision({"model-a": {"error": "timeout"}}, MODELS)
        self.assertEqual(disagreement["status"], "disagreement")
        self.assertEqual(error["status"], "error")

    def test_validates_openai_response_label(self):
        content = extract_content({"choices": [{"message": {"content": '{"category":"automated_email","confidence":0.7}'}}]})
        self.assertEqual(validate_label(json.loads(content)), {"category": "automated_email", "confidence": 0.7})
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
            "consensus_category": "business_email",
            "consensus_confidence": 0.85,
            "model_labels": {"model-a": {}, "model-b": {}},
        }
        item = build_training_item(row, annotation, max_body_chars=100)
        self.assertEqual(item["labels"]["category"], "business_email")
        self.assertEqual(item["category_label"], "business_email")
        self.assertEqual(json.loads(item["messages"][-1]["content"])["category"], "business_email")
        self.assertEqual(item["body_text"], "The service is broken")

    def test_source_weights_raise_maildir_quota(self):
        allocations = _weighted_allocations([1000, 1000, 1000], (1.0, 1.0, 3.0), 100)
        self.assertEqual(allocations, [20, 20, 60])
        validate_input_weights([Path("a"), Path("b"), Path("c")], (1.0, 1.0, 3.0))
        with self.assertRaises(SystemExit):
            validate_input_weights([Path("a")], (1.0, 2.0))


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
