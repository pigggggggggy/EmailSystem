from __future__ import annotations

import json
import re
import time

from .base import GenerationResult


class MockLLMClient:
    """Deterministic local stand-in for Qwen while the workflow is being built."""

    def generate(self, prompt: str, *, task: str, max_tokens: int = 512) -> GenerationResult:
        start = time.perf_counter()
        text = self._dispatch(prompt, task)
        latency_ms = (time.perf_counter() - start) * 1000
        return GenerationResult(
            text=text,
            input_tokens=self._rough_tokens(prompt),
            output_tokens=self._rough_tokens(text),
            latency_ms=latency_ms,
        )

    def _dispatch(self, prompt: str, task: str) -> str:
        lowered = prompt.lower()
        if task == "classify_email":
            category = "other"
            priority = "normal"
            if any(word in lowered for word in ["invoice", "payment", "receipt", "发票", "付款"]):
                category = "invoice"
            elif any(word in lowered for word in ["login", "error", "issue", "support", "无法", "故障"]):
                category = "support"
                priority = "high"
            elif any(word in lowered for word in ["meeting", "calendar", "schedule", "会议", "日程"]):
                category = "meeting"
            elif any(word in lowered for word in ["demo", "pricing", "quote", "报价", "销售"]):
                category = "sales"
            if any(word in lowered for word in ["urgent", "asap", "immediately", "紧急", "尽快"]):
                priority = "urgent"
            return json.dumps({"category": category, "priority": priority, "confidence": 0.82}, ensure_ascii=False)
        if task == "summarize_email":
            body = prompt.split("\n\n", 1)[-1].strip().replace("\n", " ")
            summary = body[:110].strip()
            if len(body) > 110:
                summary += "..."
            return json.dumps({"summary": summary or "空邮件内容。", "confidence": 0.78}, ensure_ascii=False)
        if task == "extract_action_items":
            actions = []
            if re.search(r"\b(please|need|review|send|check|confirm)\b|请|需要|确认|检查", lowered):
                actions.append({"owner": None, "task": "跟进邮件中请求的事项", "due": None})
            return json.dumps({"action_items": actions}, ensure_ascii=False)
        if task == "draft_reply":
            return "您好，邮件已收到。我们会根据内容尽快跟进，并在有进展后回复您。"
        return "{}"

    def _rough_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)
