from __future__ import annotations

import json

from email_system.models import LLMClient
from email_system.schemas import Email


class SummarizeEmailSkill:
    name = "summarize_email"

    def run(self, email: Email, context: dict, llm: LLMClient) -> dict:
        result = llm.generate(email.to_prompt_text(), task=self.name, max_tokens=256)
        data = json.loads(result.text)
        data["usage"] = result.__dict__
        return data
