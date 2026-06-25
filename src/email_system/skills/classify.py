from __future__ import annotations

from email_system.models import LLMClient
from email_system.schemas import Email
from email_system.skills.json_utils import parse_json_object


class ClassifyEmailSkill:
    name = "classify_email"

    def run(self, email: Email, context: dict, llm: LLMClient) -> dict:
        result = llm.generate(email.to_prompt_text(), task=self.name, max_tokens=128)
        data = parse_json_object(result.text)
        data["usage"] = result.__dict__
        return data
