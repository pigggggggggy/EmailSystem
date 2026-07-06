import unittest

from email_system.models import GenerationResult
from email_system.schemas import Email
from email_system.skills import ClassifyEmailSkill, ExtractActionItemsSkill, SummarizeEmailSkill


class LowConfidenceLLM:
    def generate(self, prompt: str, *, task: str, max_tokens: int = 512) -> GenerationResult:
        return GenerationResult(
            text='{"category":"automated_email","priority":"normal","confidence":0.0}',
            input_tokens=10,
            output_tokens=10,
            latency_ms=1.0,
        )


class EmptyOutputLLM:
    def generate(self, prompt: str, *, task: str, max_tokens: int = 512) -> GenerationResult:
        return GenerationResult(text="", input_tokens=10, output_tokens=0, latency_ms=1.0)


class SkillFallbackTest(unittest.TestCase):
    def setUp(self):
        self.email = Email.from_dict(
            {
                "email_id": "e-1",
                "subject": "Login issue",
                "from": "customer@example.com",
                "to": ["support@example.com"],
                "body_text": "Please check my login issue.",
            }
        )
        self.llm = EmptyOutputLLM()

    def test_classify_falls_back_on_empty_output(self):
        result = ClassifyEmailSkill().run(self.email, {}, self.llm)
        self.assertEqual(result["category"], "automated_email")
        self.assertEqual(result["priority"], "normal")
        self.assertIn("parse_error", result)

    def test_zero_confidence_is_valid_but_marked_for_review(self):
        result = ClassifyEmailSkill().run(self.email, {}, LowConfidenceLLM())

        self.assertNotIn("parse_error", result)
        self.assertEqual(result["confidence"], 0.0)
        self.assertTrue(result["low_confidence"])

    def test_summarize_falls_back_on_empty_output(self):
        result = SummarizeEmailSkill().run(self.email, {}, self.llm)
        self.assertEqual(result["summary"], "Please check my login issue.")
        self.assertIn("parse_error", result)

    def test_action_items_falls_back_on_empty_output(self):
        result = ExtractActionItemsSkill().run(self.email, {}, self.llm)
        self.assertEqual(result["action_items"], [])
        self.assertIn("parse_error", result)


if __name__ == "__main__":
    unittest.main()
