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
