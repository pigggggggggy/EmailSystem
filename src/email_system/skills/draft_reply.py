from __future__ import annotations

from email_system.models import LLMClient
from email_system.schemas import Email


class DraftReplySkill:
    name = "draft_reply"

    def run(self, email: Email, context: dict, llm: LLMClient) -> dict:
        prompt = email.to_prompt_text() + "\n\nContext: " + str(context)
        result = llm.generate(prompt, task=self.name, max_tokens=512)
        return {"reply_draft": result.text, "usage": result.__dict__}
