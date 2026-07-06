from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

from email_system.memory import MemoryRecord, ShortTermMemory
from email_system.memory.long_term import LongTermMemory
from email_system.models import LLMClient
from email_system.skills import Skill

from .state import WorkflowState, WorkflowTraceEvent


class WorkflowNode(Protocol):
    name: str

    def run(self, state: WorkflowState) -> None:
        ...


@dataclass
class LoadMemoryNode:
    short_term_memory: ShortTermMemory
    long_term_memory: LongTermMemory | None = None
    name: str = "load_memory"

    def run(self, state: WorkflowState) -> None:
        short_term = self.short_term_memory.load(state.email)
        long_term = []
        if self.long_term_memory is not None:
            long_term = [record.to_dict() for record in self.long_term_memory.search(state.email)]
        state.context["memory"] = {"short_term": short_term, "long_term": long_term}
        state.outputs[self.name] = state.context["memory"]


@dataclass
class SkillNode:
    skill: Skill
    llm: LLMClient

    @property
    def name(self) -> str:
        return self.skill.name

    def run(self, state: WorkflowState) -> None:
        output = self.skill.run(state.email, state.context, self.llm)
        state.set_output(self.name, output)


@dataclass
class HumanReviewPolicyNode:
    name: str = "human_review_policy"

    def run(self, state: WorkflowState) -> None:
        classify = state.outputs.get("classify_email", {})
        priority = classify.get("priority", "normal")
        skill_errors = {
            name: str(output["parse_error"])
            for name, output in state.outputs.items()
            if isinstance(output, dict) and output.get("parse_error")
        }
        state.set_output(
            self.name,
            {
                "requires_human_review": priority in {"high", "urgent"} or bool(skill_errors),
                "skill_errors": skill_errors,
            },
        )


@dataclass
class SaveMemoryNode:
    short_term_memory: ShortTermMemory
    long_term_memory: LongTermMemory | None = None
    name: str = "save_memory"

    def run(self, state: WorkflowState) -> None:
        output = agent_output_projection(state)
        self.short_term_memory.save(state.email, output)
        if self.long_term_memory is not None:
            self.long_term_memory.save(MemoryRecord.from_agent_output(state.email, output))
        state.outputs[self.name] = {"saved": True}


@dataclass
class TimedNode:
    node: WorkflowNode

    @property
    def name(self) -> str:
        return self.node.name

    def run(self, state: WorkflowState) -> None:
        start = time.perf_counter()
        status = "ok"
        details = {}
        try:
            self.node.run(state)
        except Exception as exc:
            status = "error"
            details = {"error": str(exc)}
            raise
        finally:
            latency_ms = (time.perf_counter() - start) * 1000
            state.timings_ms[self.name] = latency_ms
            state.add_trace(WorkflowTraceEvent(node=self.name, status=status, latency_ms=latency_ms, details=details))


def agent_output_projection(state: WorkflowState) -> dict:
    classify = state.outputs.get("classify_email", {})
    summary = state.outputs.get("summarize_email", {})
    actions = state.outputs.get("extract_action_items", {})
    review = state.outputs.get("human_review_policy", {})
    return {
        "category": classify.get("category", "automated_email"),
        "priority": classify.get("priority", "normal"),
        "summary": summary.get("summary", ""),
        "action_items": actions.get("action_items", []),
        "requires_human_review": review.get("requires_human_review", False),
    }
