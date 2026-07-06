from __future__ import annotations

from typing import Any

from email_system.models import LLMClient
from email_system.schemas import Email
from email_system.skills.json_utils import ModelOutputParseError, parse_json_object


VALID_CATEGORIES = {
    "personal_email",
    "business_email",
    "internal_email",
    "marketing_email",
    "automated_email",
    "legal_formal_email",
    "educational_email",
    "social_email",
    "special_purpose_email",
    "spam",
}
DEFAULT_CATEGORY = "automated_email"
VALID_PRIORITIES = {"low", "normal", "high", "urgent"}
LOW_CONFIDENCE_THRESHOLD = 0.5


class ClassifyEmailSkill:
    name = "classify_email"

    def run(self, email: Email, context: dict, llm: LLMClient) -> dict:
        result = llm.generate(email.to_prompt_text(), task=self.name, max_tokens=256)
        try:
            data = _validated_classification(parse_json_object(result.text), result.text)
        except ModelOutputParseError as exc:
            data = {
                "category": DEFAULT_CATEGORY,
                "priority": "normal",
                "confidence": 0.0,
                "parse_error": str(exc),
                "raw_model_output": result.text,
            }
        data["low_confidence"] = data["confidence"] < LOW_CONFIDENCE_THRESHOLD
        data["usage"] = result.__dict__
        return data


def _validated_classification(data: dict[str, Any], raw_output: str) -> dict[str, Any]:
    errors = []
    category = data.get("category")
    priority = data.get("priority")
    confidence = data.get("confidence")

    if category not in VALID_CATEGORIES:
        errors.append(f"invalid category: {category!r}")
    if priority not in VALID_PRIORITIES:
        errors.append(f"invalid priority: {priority!r}")
    valid_confidence = (
        not isinstance(confidence, bool)
        and isinstance(confidence, (int, float))
        and 0 <= confidence <= 1
    )
    if not valid_confidence:
        errors.append(f"invalid confidence: {confidence!r}")

    if errors:
        return {
            "category": category if category in VALID_CATEGORIES else DEFAULT_CATEGORY,
            "priority": priority if priority in VALID_PRIORITIES else "normal",
            "confidence": float(confidence) if valid_confidence else 0.0,
            "parse_error": "; ".join(errors),
            "raw_model_output": raw_output,
        }
    return {
        "category": str(category),
        "priority": str(priority),
        "confidence": float(confidence),
    }
