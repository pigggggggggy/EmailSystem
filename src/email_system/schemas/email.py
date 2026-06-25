from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EmailLabels:
    category: str | None = None
    priority: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "EmailLabels":
        data = data or {}
        return cls(category=data.get("category"), priority=data.get("priority"))

    def to_dict(self) -> dict[str, Any]:
        return {"category": self.category, "priority": self.priority}


@dataclass(frozen=True)
class Email:
    email_id: str
    thread_id: str | None
    subject: str
    sender: str
    to: list[str] = field(default_factory=list)
    cc: list[str] = field(default_factory=list)
    timestamp: str | None = None
    body_text: str = ""
    attachments: list[dict[str, Any]] = field(default_factory=list)
    labels: EmailLabels = field(default_factory=EmailLabels)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Email":
        return cls(
            email_id=str(data["email_id"]),
            thread_id=data.get("thread_id"),
            subject=data.get("subject", ""),
            sender=data.get("from") or data.get("sender", ""),
            to=list(data.get("to", [])),
            cc=list(data.get("cc", [])),
            timestamp=data.get("timestamp"),
            body_text=data.get("body_text", ""),
            attachments=list(data.get("attachments", [])),
            labels=EmailLabels.from_dict(data.get("labels")),
        )

    def to_prompt_text(self) -> str:
        return (
            f"Subject: {self.subject}\n"
            f"From: {self.sender}\n"
            f"To: {', '.join(self.to)}\n"
            f"Time: {self.timestamp or 'unknown'}\n\n"
            f"{self.body_text}"
        )
