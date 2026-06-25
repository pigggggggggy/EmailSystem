import unittest

from email_system.agent import EmailAgentWorkflow
from email_system.mcp import NoopMailMCPClient
from email_system.memory import InMemoryLongTermMemory, ShortTermMemory
from email_system.models import MockLLMClient
from email_system.schemas import Email


def email(email_id, subject="Urgent login issue", body="Please check this ASAP. I cannot login."):
    return Email.from_dict(
        {
            "email_id": email_id,
            "thread_id": "thread-1",
            "subject": subject,
            "from": "customer@example.com",
            "to": ["support@example.com"],
            "body_text": body,
        }
    )


class LangGraphWorkflowTest(unittest.TestCase):
    def test_support_email_routes_to_bug_tracking(self):
        workflow = EmailAgentWorkflow(
            MockLLMClient(),
            short_term_memory=ShortTermMemory(),
            long_term_memory=InMemoryLongTermMemory(),
            mail_client=NoopMailMCPClient(),
        )

        output = workflow.run(email("e-1"))
        nodes = [event["node"] for event in output.workflow_trace]

        self.assertEqual(nodes, ["read_email", "classify_intent", "bug_tracking", "write_response", "human_review", "send_reply"])
        self.assertEqual(output.category, "support")
        self.assertTrue(output.requires_human_review)
        self.assertEqual(output.memory["short_term"]["email_ids"], [])
        self.assertIn("classify_intent", output.timings_ms)

    def test_non_support_email_routes_to_documentation_search(self):
        workflow = EmailAgentWorkflow(MockLLMClient(), short_term_memory=ShortTermMemory(), long_term_memory=InMemoryLongTermMemory())

        output = workflow.run(email("e-2", subject="Invoice for June", body="Attached is the invoice for June services."))
        nodes = [event["node"] for event in output.workflow_trace]

        self.assertIn("search_documentation", nodes)
        self.assertNotIn("bug_tracking", nodes)
        self.assertEqual(output.category, "invoice")


if __name__ == "__main__":
    unittest.main()
