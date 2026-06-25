from __future__ import annotations

import time

from email_system.models import LLMClient
from email_system.schemas import ActionItem, AgentOutput, Confidence, Email, Entities
from email_system.skills import (
    ClassifyEmailSkill,
    DraftReplySkill,
    ExtractActionItemsSkill,
    SummarizeEmailSkill,
)


class EmailAgentWorkflow:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm
        self.classifier = ClassifyEmailSkill()
        self.summarizer = SummarizeEmailSkill()
        self.action_extractor = ExtractActionItemsSkill()
        self.reply_drafter = DraftReplySkill()

    def run(self, email: Email) -> AgentOutput:
        context: dict = {}
        timings_ms: dict[str, float] = {}

        classify = self._timed(self.classifier.name, timings_ms, self.classifier.run, email, context, self.llm)
        context.update(classify=classify)

        summary = self._timed(self.summarizer.name, timings_ms, self.summarizer.run, email, context, self.llm)
        context.update(summary=summary)

        actions = self._timed(
            self.action_extractor.name,
            timings_ms,
            self.action_extractor.run,
            email,
            context,
            self.llm,
        )
        context.update(actions=actions)

        reply = self._timed(self.reply_drafter.name, timings_ms, self.reply_drafter.run, email, context, self.llm)

        priority = classify.get("priority", "normal")
        skill_errors = self._skill_errors(
            classify_email=classify,
            summarize_email=summary,
            extract_action_items=actions,
        )
        return AgentOutput(
            email_id=email.email_id,
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
            requires_human_review=priority in {"high", "urgent"},
            timings_ms=timings_ms,
            skill_errors=skill_errors,
        )

    def _skill_errors(self, **outputs: dict) -> dict[str, str]:
        return {name: str(output["parse_error"]) for name, output in outputs.items() if output.get("parse_error")}

    def _timed(self, name: str, timings_ms: dict[str, float], fn, *args):
        start = time.perf_counter()
        output = fn(*args)
        timings_ms[name] = (time.perf_counter() - start) * 1000
        return output
