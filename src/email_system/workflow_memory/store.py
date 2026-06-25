from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from .pattern import WorkflowPattern


class WorkflowMemoryStore(Protocol):
    def search(self, *, category: str, subject: str, sender: str, limit: int = 3) -> list[WorkflowPattern]:
        ...

    def upsert(self, pattern: WorkflowPattern) -> None:
        ...


class InMemoryWorkflowMemoryStore:
    def __init__(self) -> None:
        self.patterns: dict[str, WorkflowPattern] = {}

    def search(self, *, category: str, subject: str, sender: str, limit: int = 3) -> list[WorkflowPattern]:
        scored = sorted(
            ((pattern.score_for(category=category, subject=subject, sender=sender), pattern) for pattern in self.patterns.values()),
            key=lambda item: item[0],
            reverse=True,
        )
        return [pattern for score, pattern in scored if score > 0][:limit]

    def upsert(self, pattern: WorkflowPattern) -> None:
        existing = self.patterns.get(pattern.workflow_id)
        if existing is None:
            self.patterns[pattern.workflow_id] = pattern
            return
        existing.success_count += pattern.success_count
        existing.source_email_ids = sorted(set(existing.source_email_ids + pattern.source_email_ids))
        senders = set(existing.metadata.get("senders", [])) | set(pattern.metadata.get("senders", []))
        existing.metadata["senders"] = sorted(senders)


class JsonlWorkflowMemoryStore:
    def __init__(self, path: str | Path = "data/memory/workflows.jsonl") -> None:
        self.path = Path(path)

    def search(self, *, category: str, subject: str, sender: str, limit: int = 3) -> list[WorkflowPattern]:
        scored = sorted(
            ((pattern.score_for(category=category, subject=subject, sender=sender), pattern) for pattern in self._read_all()),
            key=lambda item: item[0],
            reverse=True,
        )
        return [pattern for score, pattern in scored if score > 0][:limit]

    def upsert(self, pattern: WorkflowPattern) -> None:
        patterns = {item.workflow_id: item for item in self._read_all()}
        existing = patterns.get(pattern.workflow_id)
        if existing is None:
            patterns[pattern.workflow_id] = pattern
        else:
            existing.success_count += pattern.success_count
            existing.source_email_ids = sorted(set(existing.source_email_ids + pattern.source_email_ids))
            senders = set(existing.metadata.get("senders", [])) | set(pattern.metadata.get("senders", []))
            existing.metadata["senders"] = sorted(senders)
        self._write_all(patterns.values())

    def _read_all(self) -> list[WorkflowPattern]:
        if not self.path.exists():
            return []
        patterns = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    patterns.append(WorkflowPattern.from_dict(json.loads(line)))
        return patterns

    def _write_all(self, patterns) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            for pattern in patterns:
                handle.write(json.dumps(pattern.to_dict(), ensure_ascii=False) + "\n")
