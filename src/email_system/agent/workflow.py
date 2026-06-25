from __future__ import annotations

from email_system.memory import InMemoryLongTermMemory, ShortTermMemory
from email_system.memory.long_term import LongTermMemory
from email_system.models import LLMClient
from email_system.schemas import ActionItem, AgentOutput, Confidence, Email, Entities
from email_system.skills import (
    ClassifyEmailSkill,
    DraftReplySkill,
    ExtractActionItemsSkill,
    SummarizeEmailSkill,
)

from .nodes import HumanReviewPolicyNode, LoadMemoryNode, SaveMemoryNode, SkillNode, TimedNode
from .state import WorkflowState


class EmailAgentWorkflow:
    def __init__(
        self,
        llm: LLMClient,
        *,
        short_term_memory: ShortTermMemory | None = None,
        long_term_memory: LongTermMemory | None = None,
    ) -> None:
        self.llm = llm
        self.short_term_memory = short_term_memory or ShortTermMemory()
        self.long_term_memory = long_term_memory if long_term_memory is not None else InMemoryLongTermMemory()
        self.nodes = [
            TimedNode(LoadMemoryNode(self.short_term_memory, self.long_term_memory)),
            TimedNode(SkillNode(ClassifyEmailSkill(), self.llm)),
            TimedNode(SkillNode(SummarizeEmailSkill(), self.llm)),
            TimedNode(SkillNode(ExtractActionItemsSkill(), self.llm)),
            TimedNode(SkillNode(DraftReplySkill(), self.llm)),
            TimedNode(HumanReviewPolicyNode()),
            TimedNode(SaveMemoryNode(self.short_term_memory, self.long_term_memory)),
        ]

    def run(self, email: Email) -> AgentOutput:
        state = WorkflowState(email=email)
        for node in self.nodes:
            node.run(state)
        return self._build_output(state)

    def _build_output(self, state: WorkflowState) -> AgentOutput:
        classify = state.outputs.get("classify_email", {})
        summary = state.outputs.get("summarize_email", {})
        actions = state.outputs.get("extract_action_items", {})
        reply = state.outputs.get("draft_reply", {})
        review = state.outputs.get("human_review_policy", {})
        priority = classify.get("priority", "normal")
        return AgentOutput(
            email_id=state.email.email_id,
            category=classify.get("category", "other"),
            priority=priority,
            summary=summary.get("summary", ""),
            action_items=[ActionItem(**item) for item in actions.get("action_items", [])],
            entities=Entities(),
            reply_draft=reply.get("reply_draft", ""),
            confidence=Confidence(
                category=float(classify.get("confidence", 0.0)),
                summary=float(summary.get("confidence", 0.0)),
            ),
            requires_human_review=review.get("requires_human_review", priority in {"high", "urgent"}),
            timings_ms=state.timings_ms,
            skill_errors=review.get("skill_errors", {}),
            memory=state.outputs.get("load_memory", {}),
            workflow_trace=[event.to_dict() for event in state.trace],
        )
