from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class WorkflowStep:
    observation: str
    reasoning: str
    action: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkflowStep":
        return cls(
            observation=str(data.get("observation", "")),
            reasoning=str(data.get("reasoning", "")),
            action=str(data.get("action", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WorkflowPattern:
    workflow_id: str
    description: str
    category: str
    priority: str
    steps: list[WorkflowStep]
    source_email_ids: list[str] = field(default_factory=list)
    success_count: int = 1
    use_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkflowPattern":
        return cls(
            workflow_id=str(data["workflow_id"]),
            description=str(data.get("description", "")),
            category=str(data.get("category", "other")),
            priority=str(data.get("priority", "normal")),
            steps=[WorkflowStep.from_dict(item) for item in data.get("steps", [])],
            source_email_ids=list(data.get("source_email_ids", [])),
            success_count=int(data.get("success_count", 1)),
            use_count=int(data.get("use_count", 0)),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["steps"] = [step.to_dict() for step in self.steps]
        return data

    def score_for(self, *, category: str, subject: str, sender: str) -> int:
        score = 0
        if self.category == category:
            score += 8
        subject_terms = set(subject.lower().split())
        description_terms = set(self.description.lower().split())
        score += len(subject_terms & description_terms)
        if sender and sender in self.metadata.get("senders", []):
            score += 2
        score += min(self.success_count, 5)
        return score
