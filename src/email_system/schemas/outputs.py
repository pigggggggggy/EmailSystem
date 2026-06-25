from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ActionItem:
    owner: str | None
    task: str
    due: str | None = None


@dataclass(frozen=True)
class Entities:
    people: list[str] = field(default_factory=list)
    companies: list[str] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)
    order_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Confidence:
    category: float = 0.0
    summary: float = 0.0


@dataclass(frozen=True)
class AgentOutput:
    email_id: str
    category: str
    priority: str
    summary: str
    action_items: list[ActionItem] = field(default_factory=list)
    entities: Entities = field(default_factory=Entities)
    reply_draft: str = ""
    confidence: Confidence = field(default_factory=Confidence)
    requires_human_review: bool = False
    timings_ms: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
