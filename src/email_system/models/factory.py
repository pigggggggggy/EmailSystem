from __future__ import annotations

from pathlib import Path

from .base import LLMClient
from .mock import MockLLMClient
from .transformers_client import TransformersLLMClient


def build_llm_client(
    backend: str,
    *,
    model_path: str | Path = "models/Qwen3-4B",
    device_map: str = "auto",
    torch_dtype: str = "auto",
) -> LLMClient:
    if backend == "mock":
        return MockLLMClient()
    if backend == "transformers":
        return TransformersLLMClient(model_path, device_map=device_map, torch_dtype=torch_dtype)
    raise ValueError(f"Unsupported backend: {backend}")
