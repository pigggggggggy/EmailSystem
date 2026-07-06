import unittest

from email_system.agent import EmailAgentWorkflow
from email_system.models import MockLLMClient
from email_system.schemas import Email


class WorkflowTest(unittest.TestCase):
    def test_workflow_outputs_structured_result(self):
        email = Email.from_dict(
            {
                "email_id": "t-1",
                "thread_id": "th-1",
                "subject": "Urgent login issue",
                "from": "customer@example.com",
                "to": ["support@example.com"],
                "body_text": "Please check this ASAP. I cannot login.",
                "labels": {"category": "business_email", "priority": "urgent"},
            }
        )

        output = EmailAgentWorkflow(MockLLMClient()).run(email)

        self.assertEqual(output.email_id, "t-1")
        self.assertEqual(output.category, "business_email")
        self.assertEqual(output.priority, "urgent")
        self.assertTrue(output.requires_human_review)
        self.assertTrue(output.summary)
        self.assertIn("classify_intent", output.timings_ms)


if __name__ == "__main__":
    unittest.main()
