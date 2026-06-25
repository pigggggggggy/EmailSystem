from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol

from email_system.schemas import Email


@dataclass(frozen=True)
class MailMCPResult:
    action: str
    status: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MailMCPClient(Protocol):
    def read_email(self, email: Email) -> MailMCPResult:
        ...

    def bug_tracking(self, email: Email, context: dict[str, Any]) -> MailMCPResult:
        ...

    def search_documentation(self, email: Email, context: dict[str, Any]) -> MailMCPResult:
        ...

    def send_reply(self, email: Email, reply_text: str, context: dict[str, Any]) -> MailMCPResult:
        ...


class NoopMailMCPClient:
    """Local stand-in for the mcp_agent_mail MCP server.

    The external project exposes a FastMCP mail-like coordination layer with
    identities, inbox/outbox, searchable threads, and message sending. This
    adapter keeps our workflow independent from the transport so tests and mock
    runs do not require a running MCP server.
    """

    def read_email(self, email: Email) -> MailMCPResult:
        return MailMCPResult(
            action="read_email",
            status="local",
            metadata={"email_id": email.email_id, "thread_id": email.thread_id or email.email_id},
        )

    def bug_tracking(self, email: Email, context: dict[str, Any]) -> MailMCPResult:
        return MailMCPResult(
            action="bug_tracking",
            status="prepared",
            metadata={"email_id": email.email_id, "suggested_queue": "support_bug_triage"},
        )

    def search_documentation(self, email: Email, context: dict[str, Any]) -> MailMCPResult:
        return MailMCPResult(
            action="search_documentation",
            status="prepared",
            metadata={"email_id": email.email_id, "top_k": 3},
        )

    def send_reply(self, email: Email, reply_text: str, context: dict[str, Any]) -> MailMCPResult:
        requires_review = bool(context.get("human_review", {}).get("requires_human_review", True))
        return MailMCPResult(
            action="send_reply",
            status="pending_human_review" if requires_review else "ready_to_send",
            metadata={"email_id": email.email_id, "reply_chars": len(reply_text)},
        )


class GmailMailMCPClient:
    """Gmail-backed implementation of the workflow mail interface."""

    def __init__(self, gmail_client, *, send_mode: str = "dry_run") -> None:
        if send_mode not in {"dry_run", "draft", "send"}:
            raise ValueError("send_mode must be one of: dry_run, draft, send")
        self.gmail_client = gmail_client
        self.send_mode = send_mode

    def read_email(self, email: Email) -> MailMCPResult:
        return MailMCPResult(
            action="read_email",
            status="gmail_loaded",
            metadata={"email_id": email.email_id, "thread_id": email.thread_id or email.email_id},
        )

    def bug_tracking(self, email: Email, context: dict[str, Any]) -> MailMCPResult:
        return MailMCPResult(
            action="bug_tracking",
            status="prepared",
            metadata={"email_id": email.email_id, "source": "gmail", "suggested_queue": "support_bug_triage"},
        )

    def search_documentation(self, email: Email, context: dict[str, Any]) -> MailMCPResult:
        return MailMCPResult(
            action="search_documentation",
            status="prepared",
            metadata={"email_id": email.email_id, "source": "gmail", "top_k": 3},
        )

    def send_reply(self, email: Email, reply_text: str, context: dict[str, Any]) -> MailMCPResult:
        if self.send_mode == "dry_run":
            return MailMCPResult(
                action="send_reply",
                status="dry_run",
                metadata={"email_id": email.email_id, "reply_chars": len(reply_text)},
            )
        if self.send_mode == "draft":
            draft = self.gmail_client.create_draft_reply(email=email, reply_text=reply_text)
            return MailMCPResult(
                action="send_reply",
                status="draft_created",
                metadata={"email_id": email.email_id, "draft_id": draft.get("id")},
            )
        sent = self.gmail_client.send_reply(email=email, reply_text=reply_text)
        return MailMCPResult(
            action="send_reply",
            status="sent",
            metadata={"email_id": email.email_id, "message_id": sent.get("id")},
        )
