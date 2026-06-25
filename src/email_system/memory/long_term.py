from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from email_system.schemas import Email


@dataclass(frozen=True)
class MemoryRecord:
    email_id: str
    thread_id: str
    subject: str
    sender: str
    category: str
    priority: str
    summary: str
    action_count: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_agent_output(cls, email: Email, output: dict[str, Any]) -> "MemoryRecord":
        return cls(
            email_id=email.email_id,
            thread_id=email.thread_id or email.email_id,
            subject=email.subject,
            sender=email.sender,
            category=output.get("category", "other"),
            priority=output.get("priority", "normal"),
            summary=output.get("summary", ""),
            action_count=len(output.get("action_items", [])),
            metadata={"requires_human_review": output.get("requires_human_review", False)},
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LongTermMemory(Protocol):
    def search(self, email: Email, *, limit: int = 3) -> list[MemoryRecord]:
        ...

    def save(self, record: MemoryRecord) -> None:
        ...


class JsonlLongTermMemory:
    """Append-only local long-term memory store."""

    def __init__(self, path: str | Path = "data/memory/long_term.jsonl") -> None:
        self.path = Path(path)

    def search(self, email: Email, *, limit: int = 3) -> list[MemoryRecord]:
        records = self._read_all()
        scored = sorted(
            ((self._score(email, record), record) for record in records),
            key=lambda item: item[0],
            reverse=True,
        )
        return [record for score, record in scored if score > 0][:limit]

    def save(self, record: MemoryRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    def _read_all(self) -> list[MemoryRecord]:
        if not self.path.exists():
            return []
        records = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(MemoryRecord(**json.loads(line)))
        return records

    def _score(self, email: Email, record: MemoryRecord) -> int:
        score = 0
        if record.thread_id == (email.thread_id or email.email_id):
            score += 5
        if record.sender == email.sender:
            score += 2
        subject_terms = set(email.subject.lower().split())
        record_terms = set(record.subject.lower().split())
        score += len(subject_terms & record_terms)
        return score
