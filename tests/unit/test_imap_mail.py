import unittest
from email.message import EmailMessage

from email_system.imap_mail import emails_to_dicts, message_to_email


class IMAPMailTest(unittest.TestCase):
    def test_message_to_email_plain_text(self):
        message = EmailMessage()
        message["Subject"] = "Login issue"
        message["From"] = "sender@example.com"
        message["To"] = "support@example.com"
        message["Message-ID"] = "<msg-1@example.com>"
        message.set_content("Please check login.")

        item = message_to_email(message, imap_id="1")

        self.assertEqual(item.email_id, "<msg-1@example.com>")
        self.assertEqual(item.subject, "Login issue")
        self.assertEqual(item.sender, "sender@example.com")
        self.assertEqual(item.to, ["support@example.com"])
        self.assertIn("Please check login.", item.body_text)

    def test_emails_to_dicts_preserves_sender_key(self):
        message = EmailMessage()
        message["Subject"] = "Hello"
        message["From"] = "sender@example.com"
        message.set_content("Hello")
        item = message_to_email(message, imap_id="1")

        rows = emails_to_dicts([item])

        self.assertEqual(rows[0]["from"], "sender@example.com")
        self.assertIn("body_text", rows[0])


if __name__ == "__main__":
    unittest.main()
