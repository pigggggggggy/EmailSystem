from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class GenerationResult:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


class LLMClient(Protocol):
    def generate(self, prompt: str, *, task: str, max_tokens: int = 512) -> GenerationResult:
        ...
