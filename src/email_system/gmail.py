from __future__ import annotations

import base64
from email.message import EmailMessage
from pathlib import Path
from typing import Any

from email_system.schemas import Email

GMAIL_READONLY_SCOPE = "https://www.googleapis.com/auth/gmail.readonly"
GMAIL_COMPOSE_SCOPE = "https://www.googleapis.com/auth/gmail.compose"
GMAIL_SEND_SCOPE = "https://www.googleapis.com/auth/gmail.send"


def build_gmail_service(
    *,
    credentials_path: str | Path = "secrets/gmail_credentials.json",
    token_path: str | Path = "data/auth/gmail_token.json",
    scopes: list[str] | None = None,
):
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise RuntimeError("Gmail support requires: pip install -e '.[gmail]'") from exc

    scopes = scopes or [GMAIL_READONLY_SCOPE]
    token = Path(token_path)
    credentials = Path(credentials_path)
    creds = None
    if token.exists():
        creds = Credentials.from_authorized_user_file(str(token), scopes)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not credentials.exists():
                raise FileNotFoundError(f"Missing Gmail OAuth client credentials: {credentials}")
            flow = InstalledAppFlow.from_client_secrets_file(str(credentials), scopes)
            creds = flow.run_local_server(port=0)
        token.parent.mkdir(parents=True, exist_ok=True)
        token.write_text(creds.to_json(), encoding="utf-8")
    return build("gmail", "v1", credentials=creds)


class GmailAPIClient:
    def __init__(self, service) -> None:
        self.service = service

    def list_emails(self, *, query: str = "in:inbox newer_than:30d", limit: int = 10) -> list[Email]:
        response = self.service.users().messages().list(userId="me", q=query, maxResults=limit).execute()
        messages = response.get("messages", [])
        emails = []
        for message in messages:
            raw = self.service.users().messages().get(userId="me", id=message["id"], format="full").execute()
            emails.append(self._message_to_email(raw))
        return emails

    def create_draft_reply(self, *, email: Email, reply_text: str) -> dict[str, Any]:
        mime = EmailMessage()
        mime.set_content(reply_text)
        mime["To"] = email.sender
        mime["Subject"] = _reply_subject(email.subject)
        encoded = base64.urlsafe_b64encode(mime.as_bytes()).decode()
        body = {"message": {"raw": encoded, "threadId": email.thread_id or None}}
        return self.service.users().drafts().create(userId="me", body=body).execute()

    def send_reply(self, *, email: Email, reply_text: str) -> dict[str, Any]:
        mime = EmailMessage()
        mime.set_content(reply_text)
        mime["To"] = email.sender
        mime["Subject"] = _reply_subject(email.subject)
        encoded = base64.urlsafe_b64encode(mime.as_bytes()).decode()
        body = {"raw": encoded, "threadId": email.thread_id or None}
        return self.service.users().messages().send(userId="me", body=body).execute()

    def _message_to_email(self, message: dict[str, Any]) -> Email:
        payload = message.get("payload", {})
        headers = _headers(payload)
        return Email.from_dict(
            {
                "email_id": message["id"],
                "thread_id": message.get("threadId"),
                "subject": headers.get("subject", ""),
                "from": headers.get("from", ""),
                "to": _split_addresses(headers.get("to", "")),
                "cc": _split_addresses(headers.get("cc", "")),
                "timestamp": _internal_date(message),
                "body_text": _payload_text(payload),
                "attachments": _attachments(payload),
                "labels": {},
            }
        )


def _headers(payload: dict[str, Any]) -> dict[str, str]:
    return {header.get("name", "").lower(): header.get("value", "") for header in payload.get("headers", [])}


def _payload_text(payload: dict[str, Any]) -> str:
    parts = _walk_parts(payload)
    plain = [part for part in parts if part.get("mimeType") == "text/plain"]
    html = [part for part in parts if part.get("mimeType") == "text/html"]
    for part in plain or html or [payload]:
        data = part.get("body", {}).get("data")
        if data:
            return _decode_body(data)
    return ""


def _walk_parts(payload: dict[str, Any]) -> list[dict[str, Any]]:
    parts = []
    stack = list(payload.get("parts", []))
    while stack:
        part = stack.pop(0)
        parts.append(part)
        stack.extend(part.get("parts", []))
    return parts


def _decode_body(data: str) -> str:
    padded = data + "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(padded.encode("utf-8")).decode("utf-8", errors="replace")


def _attachments(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"filename": part.get("filename", ""), "mime_type": part.get("mimeType", "")}
        for part in _walk_parts(payload)
        if part.get("filename")
    ]


def _split_addresses(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _internal_date(message: dict[str, Any]) -> str | None:
    value = message.get("internalDate")
    return str(value) if value is not None else None


def _reply_subject(subject: str) -> str:
    return subject if subject.lower().startswith("re:") else f"Re: {subject}"
