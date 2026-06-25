from __future__ import annotations

from email_system.models import LLMClient
from email_system.schemas import Email
from email_system.skills.json_utils import ModelOutputParseError, parse_json_object


class ExtractActionItemsSkill:
    name = "extract_action_items"

    def run(self, email: Email, context: dict, llm: LLMClient) -> dict:
        result = llm.generate(email.to_prompt_text(), task=self.name, max_tokens=256)
        try:
            data = parse_json_object(result.text)
        except ModelOutputParseError as exc:
            data = {
                "action_items": [],
                "parse_error": str(exc),
                "raw_model_output": result.text,
            }
        data["usage"] = result.__dict__
        return data
