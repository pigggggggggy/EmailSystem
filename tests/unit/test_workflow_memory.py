import unittest

from email_system.agent import EmailAgentWorkflow
from email_system.memory import InMemoryLongTermMemory, ShortTermMemory
from email_system.models import MockLLMClient
from email_system.schemas import Email


def email(email_id, body="Please check this login issue."):
    return Email.from_dict(
        {
            "email_id": email_id,
            "thread_id": "thread-1",
            "subject": "Login issue",
            "from": "customer@example.com",
            "to": ["support@example.com"],
            "body_text": body,
        }
    )


class WorkflowMemoryTest(unittest.TestCase):
    def test_workflow_loads_and_saves_memory(self):
        short_term = ShortTermMemory()
        long_term = InMemoryLongTermMemory()
        workflow = EmailAgentWorkflow(MockLLMClient(), short_term_memory=short_term, long_term_memory=long_term)

        first = workflow.run(email("e-1"))
        second = workflow.run(email("e-2"))

        self.assertEqual(first.memory["short_term"]["email_ids"], [])
        self.assertEqual(second.memory["short_term"]["email_ids"], ["e-1"])
        self.assertGreaterEqual(len(second.memory["long_term"]), 1)
        self.assertIn("load_memory", second.timings_ms)
        self.assertEqual(second.workflow_trace[0]["node"], "load_memory")
        self.assertEqual(second.workflow_trace[-1]["node"], "save_memory")


if __name__ == "__main__":
    unittest.main()
