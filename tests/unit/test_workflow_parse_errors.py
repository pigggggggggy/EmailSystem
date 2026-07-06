import unittest

from email_system.agent import EmailAgentWorkflow
from email_system.models import GenerationResult
from email_system.schemas import Email


class EmptyJsonLLM:
    def generate(self, prompt: str, *, task: str, max_tokens: int = 512) -> GenerationResult:
        text = "您好，邮件已收到。" if task == "draft_reply" else ""
        return GenerationResult(text=text, input_tokens=10, output_tokens=0, latency_ms=1.0)


class WorkflowParseErrorsTest(unittest.TestCase):
    def test_workflow_reports_nonfatal_skill_errors(self):
        email = Email.from_dict(
            {
                "email_id": "e-1",
                "subject": "Login issue",
                "from": "customer@example.com",
                "to": ["support@example.com"],
                "body_text": "Please check my login issue.",
            }
        )

        output = EmailAgentWorkflow(EmptyJsonLLM()).run(email)

        self.assertEqual(output.category, "automated_email")
        self.assertIn("classify_intent", output.skill_errors)
        self.assertIn("summarize_email", output.skill_errors)
        self.assertIn("extract_action_items", output.skill_errors)


if __name__ == "__main__":
    unittest.main()
