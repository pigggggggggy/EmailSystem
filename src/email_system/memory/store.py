from __future__ import annotations

from email_system.schemas import Email

from .long_term import MemoryRecord


class InMemoryLongTermMemory:
    def __init__(self) -> None:
        self.records: list[MemoryRecord] = []

    def search(self, email: Email, *, limit: int = 3) -> list[MemoryRecord]:
        thread_id = email.thread_id or email.email_id
        matches = [record for record in self.records if record.thread_id == thread_id or record.sender == email.sender]
        return matches[-limit:]

    def save(self, record: MemoryRecord) -> None:
        self.records.append(record)
