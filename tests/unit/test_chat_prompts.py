import unittest

from email_system.models.chat_prompts import PROMPT_VERSION, messages_for_task


class ChatPromptsTest(unittest.TestCase):
    def test_classification_prompt_uses_real_confidence_example(self):
        prompt = messages_for_task("Subject: hello", "classify_email")[1]["content"]

        self.assertIn('"confidence": 0.85', prompt)
        self.assertNotIn("invoice|support|meeting", prompt)
        self.assertNotIn('"confidence": 0.0', prompt)
        self.assertTrue(PROMPT_VERSION)

    def test_draft_reply_prompt_requests_sendable_reply_not_summary(self):
        prompt = messages_for_task("Subject: hello", "draft_reply")[1]["content"]

        self.assertIn("可直接发送给发件人", prompt)
        self.assertIn("不要复述或总结原邮件", prompt)
        self.assertIn("不要编造事实", prompt)
        self.assertIn("80 到 180 个中文字符", prompt)
        self.assertIn("广告、陌生交友、钓鱼或垃圾邮件", prompt)



if __name__ == "__main__":
    unittest.main()
