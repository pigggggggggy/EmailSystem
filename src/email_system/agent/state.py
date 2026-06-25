from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from email_system.schemas import Email


@dataclass
class WorkflowTraceEvent:
    node: str
    status: str
    latency_ms: float
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node": self.node,
            "status": self.status,
            "latency_ms": self.latency_ms,
            "details": self.details,
        }


@dataclass
class WorkflowState:
    email: Email
    context: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    timings_ms: dict[str, float] = field(default_factory=dict)
    trace: list[WorkflowTraceEvent] = field(default_factory=list)

    def set_output(self, name: str, value: dict[str, Any]) -> None:
        self.outputs[name] = value
        self.context[name] = value

    def add_trace(self, event: WorkflowTraceEvent) -> None:
        self.trace.append(event)
