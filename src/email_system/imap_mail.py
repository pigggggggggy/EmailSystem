from __future__ import annotations

import email as email_parser
import imaplib
import smtplib
from dataclasses import dataclass
from email.header import decode_header, make_header
from email.message import EmailMessage, Message
from email.utils import getaddresses, parsedate_to_datetime
from typing import Iterable

from email_system.schemas import Email


@dataclass(frozen=True)
class IMAPConfig:
    host: str = "imap.gmail.com"
    port: int = 993
    mailbox: str = "INBOX"


@dataclass(frozen=True)
class SMTPConfig:
    host: str = "smtp.gmail.com"
    port: int = 465


class IMAPEmailClient:
    def __init__(self, *, user: str, password: str, config: IMAPConfig | None = None) -> None:
        self.user = user
        self.password = password
        self.config = config or IMAPConfig()

    def fetch_recent(self, *, limit: int = 10, search: str = "ALL") -> list[Email]:
        with imaplib.IMAP4_SSL(self.config.host, self.config.port) as client:
            client.login(self.user, self.password)
            client.select(self.config.mailbox, readonly=True)
            status, data = client.search(None, search)
            if status != "OK":
                raise RuntimeError(f"IMAP search failed: {status}")
            ids = data[0].split()[-limit:]
            emails = []
            for message_id in reversed(ids):
                status, fetched = client.fetch(message_id, "(RFC822)")
                if status != "OK" or not fetched or not isinstance(fetched[0], tuple):
                    continue
                raw = fetched[0][1]
                message = email_parser.message_from_bytes(raw)
                emails.append(message_to_email(message, imap_id=message_id.decode("ascii", errors="replace")))
            return emails


class SMTPEmailClient:
    def __init__(self, *, user: str, password: str, config: SMTPConfig | None = None) -> None:
        self.user = user
        self.password = password
        self.config = config or SMTPConfig()

    def send_reply(self, *, original: Email, reply_text: str) -> dict:
        message = EmailMessage()
        message["From"] = self.user
        message["To"] = original.sender
        message["Subject"] = _reply_subject(original.subject)
        message.set_content(reply_text)
        with smtplib.SMTP_SSL(self.config.host, self.config.port) as client:
            client.login(self.user, self.password)
            client.send_message(message)
        return {"status": "sent", "to": original.sender, "subject": message["Subject"]}


def message_to_email(message: Message, *, imap_id: str) -> Email:
    subject = _header(message.get("Subject", ""))
    sender = _header(message.get("From", ""))
    to = _addresses(message.get_all("To", []))
    cc = _addresses(message.get_all("Cc", []))
    date = _date(message.get("Date"))
    message_id = _header(message.get("Message-ID", "")) or f"imap:{imap_id}"
    thread_id = _header(message.get("In-Reply-To", "")) or message_id
    return Email.from_dict(
        {
            "email_id": message_id,
            "thread_id": thread_id,
            "subject": subject,
            "from": sender,
            "to": to,
            "cc": cc,
            "timestamp": date,
            "body_text": _body_text(message),
            "attachments": _attachments(message),
            "labels": {},
        }
    )


def emails_to_dicts(emails: Iterable[Email]) -> list[dict]:
    return [
        {
            "email_id": item.email_id,
            "thread_id": item.thread_id,
            "subject": item.subject,
            "from": item.sender,
            "to": item.to,
            "cc": item.cc,
            "timestamp": item.timestamp,
            "body_text": item.body_text,
            "attachments": item.attachments,
            "labels": item.labels.to_dict(),
        }
        for item in emails
    ]


def _header(value: str) -> str:
    try:
        return str(make_header(decode_header(value))).strip()
    except Exception:
        return value.strip()


def _addresses(values: list[str]) -> list[str]:
    return [address for _, address in getaddresses(values) if address]


def _date(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return parsedate_to_datetime(value).isoformat()
    except Exception:
        return value


def _body_text(message: Message) -> str:
    if message.is_multipart():
        plain_parts = []
        html_parts = []
        for part in message.walk():
            if part.get_content_disposition() == "attachment":
                continue
            content_type = part.get_content_type()
            if content_type not in {"text/plain", "text/html"}:
                continue
            payload = _decode_part(part)
            if content_type == "text/plain":
                plain_parts.append(payload)
            else:
                html_parts.append(payload)
        return "\n".join(plain_parts or html_parts).strip()
    return _decode_part(message).strip()


def _decode_part(part: Message) -> str:
    payload = part.get_payload(decode=True)
    if payload is None:
        raw = part.get_payload()
        return raw if isinstance(raw, str) else ""
    charset = part.get_content_charset() or "utf-8"
    return payload.decode(charset, errors="replace")


def _attachments(message: Message) -> list[dict]:
    attachments = []
    for part in message.walk() if message.is_multipart() else []:
        if part.get_content_disposition() == "attachment":
            attachments.append({"filename": _header(part.get_filename() or ""), "mime_type": part.get_content_type()})
    return attachments


def _reply_subject(subject: str) -> str:
    return subject if subject.lower().startswith("re:") else f"Re: {subject}"
