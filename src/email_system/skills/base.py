from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from email_system.models import LLMClient
from email_system.schemas import Email


class Skill(Protocol):
    name: str

    def run(self, email: Email, context: dict, llm: LLMClient) -> dict:
        ...


@dataclass(frozen=True)
class JsonSkillConfig:
    max_tokens: int = 512
