import unittest

from email_system.gmail import GmailAPIClient
from email_system.mcp import GmailMailMCPClient
from email_system.schemas import Email


class GmailClientTest(unittest.TestCase):
    def test_message_to_email_decodes_plain_text(self):
        client = GmailAPIClient(service=None)
        message = {
            "id": "msg-1",
            "threadId": "thread-1",
            "internalDate": "123456789",
            "payload": {
                "headers": [
                    {"name": "Subject", "value": "Hello"},
                    {"name": "From", "value": "sender@example.com"},
                    {"name": "To", "value": "me@example.com"},
                ],
                "parts": [{"mimeType": "text/plain", "body": {"data": "SGVsbG8="}}],
            },
        }

        email = client._message_to_email(message)

        self.assertEqual(email.email_id, "msg-1")
        self.assertEqual(email.thread_id, "thread-1")
        self.assertEqual(email.subject, "Hello")
        self.assertEqual(email.body_text, "Hello")

    def test_gmail_mcp_dry_run_does_not_send(self):
        email = Email.from_dict(
            {
                "email_id": "msg-1",
                "thread_id": "thread-1",
                "subject": "Hello",
                "from": "sender@example.com",
                "to": ["me@example.com"],
                "body_text": "Hello",
            }
        )
        client = GmailMailMCPClient(gmail_client=object(), send_mode="dry_run")

        result = client.send_reply(email, "Thanks", {})

        self.assertEqual(result.status, "dry_run")
        self.assertEqual(result.metadata["reply_chars"], 6)


if __name__ == "__main__":
    unittest.main()
