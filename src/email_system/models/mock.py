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
            lowered = lowered.rsplit("subject:", 1)[-1]
            category = "business_email"
            priority = "normal"
            if any(word in lowered for word in ["phishing", "credential", "lottery", "winner", "malware", "钓鱼", "诈骗", "中奖"]):
                category = "spam"
            elif any(word in lowered for word in ["legal notice", "violation", "formal warning", "contract signing", "法律通知", "违规", "正式警告", "合同签署"]):
                category = "legal_formal_email"
            elif any(word in lowered for word in ["course", "exam schedule", "examination", "transcript", "lecture", "课程", "考试", "成绩单", "讲座"]):
                category = "educational_email"
            elif any(word in lowered for word in ["internal announcement", "team update", "employee feedback", "内部公告", "团队进展", "员工反馈"]):
                category = "internal_email"
            elif any(word in lowered for word in ["discount", "coupon", "special offer", "promotion", "newsletter", "digest", "unsubscribe", "折扣", "优惠", "促销", "简报", "订阅"]):
                category = "marketing_email"
            elif any(word in lowered for word in ["verification code", "password reset", "invoice", "payment", "receipt", "shipping", "delivery", "welcome", "验证码", "密码重置", "发票", "收据", "物流", "欢迎"]):
                category = "automated_email"
            elif any(word in lowered for word in ["survey", "maintenance", "service interruption", "outage", "调查", "系统维护", "服务中断"]):
                category = "special_purpose_email"
            elif any(word in lowered for word in ["happy birthday", "congratulations", "invitation", "party", "生日快乐", "祝贺", "邀请", "聚会"]):
                category = "social_email"
            elif any(word in lowered for word in ["family", "friend", "travel", "thank you", "家人", "朋友", "旅行", "感谢"]):
                category = "personal_email"
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
