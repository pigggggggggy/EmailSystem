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


class LowConfidenceLLM:
    def generate(self, prompt, *, task, max_tokens=512):
        from email_system.models import GenerationResult
        import json
        outputs = {
            "classify_email": json.dumps({"category": "personal_email", "priority": "normal", "confidence": 0.0}),
            "summarize_email": json.dumps({"summary": "summary", "confidence": 0.8}),
            "extract_action_items": json.dumps({"action_items": []}),
            "draft_reply": "reply",
        }
        return GenerationResult(text=outputs[task], input_tokens=10, output_tokens=5, latency_ms=1.0)


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
        self.assertEqual(output.category, "business_email")
        self.assertTrue(output.requires_human_review)
        self.assertEqual(output.memory["short_term"]["email_ids"], [])
        self.assertIn("classify_intent", output.timings_ms)
        self.assertEqual(output.model_backend, "MockLLMClient")
        self.assertIn(output.graph_backend, {"langgraph", "fallback"})
        self.assertEqual(output.route, "bug_tracking")
        self.assertEqual(output.delivery_status, "pending_human_review")

    def test_low_confidence_requires_human_review(self):
        workflow = EmailAgentWorkflow(LowConfidenceLLM())

        output = workflow.run(email("e-low", subject="Hello", body="How are you?"))

        self.assertTrue(output.requires_human_review)
        self.assertEqual(output.delivery_status, "pending_human_review")

    def test_non_support_email_routes_to_documentation_search(self):
        workflow = EmailAgentWorkflow(MockLLMClient(), short_term_memory=ShortTermMemory(), long_term_memory=InMemoryLongTermMemory())

        output = workflow.run(email("e-2", subject="Invoice for June", body="Attached is the invoice for June services."))
        nodes = [event["node"] for event in output.workflow_trace]

        self.assertIn("search_documentation", nodes)
        self.assertNotIn("bug_tracking", nodes)
        self.assertEqual(output.category, "automated_email")
        self.assertEqual(output.route, "search_documentation")
        self.assertEqual(output.delivery_status, "ready_to_send")


if __name__ == "__main__":
    unittest.main()
