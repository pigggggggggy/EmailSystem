from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from email_system.schemas import Email


@dataclass
class ThreadMemory:
    thread_id: str
    email_ids: list[str] = field(default_factory=list)
    recent_summaries: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)

    def to_context(self) -> dict[str, Any]:
        return {
            "thread_id": self.thread_id,
            "email_ids": list(self.email_ids),
            "recent_summaries": list(self.recent_summaries),
            "categories": list(self.categories),
        }


class ShortTermMemory:
    """In-process thread memory for the current agent runtime."""

    def __init__(self, *, max_threads: int = 100, max_items_per_thread: int = 8) -> None:
        self.max_threads = max_threads
        self.max_items_per_thread = max_items_per_thread
        self._threads: OrderedDict[str, ThreadMemory] = OrderedDict()

    def load(self, email: Email) -> dict[str, Any]:
        thread_id = self._thread_id(email)
        memory = self._threads.get(thread_id)
        if memory is None:
            return {"thread_id": thread_id, "email_ids": [], "recent_summaries": [], "categories": []}
        self._threads.move_to_end(thread_id)
        return memory.to_context()

    def save(self, email: Email, output: dict[str, Any]) -> None:
        thread_id = self._thread_id(email)
        memory = self._threads.get(thread_id)
        if memory is None:
            memory = ThreadMemory(thread_id=thread_id)
            self._threads[thread_id] = memory
        self._threads.move_to_end(thread_id)
        self._append(memory.email_ids, email.email_id)
        self._append(memory.recent_summaries, output.get("summary", ""))
        self._append(memory.categories, output.get("category", "automated_email"))
        while len(self._threads) > self.max_threads:
            self._threads.popitem(last=False)

    def _append(self, values: list[str], value: str) -> None:
        if value:
            values.append(value)
        del values[:-self.max_items_per_thread]

    def _thread_id(self, email: Email) -> str:
        return email.thread_id or email.email_id
