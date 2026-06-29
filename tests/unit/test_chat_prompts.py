import unittest

from email_system.models.chat_prompts import PROMPT_VERSION, messages_for_task


class ChatPromptsTest(unittest.TestCase):
    def test_classification_prompt_uses_real_confidence_example(self):
        prompt = messages_for_task("Subject: hello", "classify_email")[1]["content"]

        self.assertIn('"confidence": 0.85', prompt)
        self.assertNotIn("invoice|support|meeting", prompt)
        self.assertNotIn('"confidence": 0.0', prompt)
        self.assertTrue(PROMPT_VERSION)


if __name__ == "__main__":
    unittest.main()
