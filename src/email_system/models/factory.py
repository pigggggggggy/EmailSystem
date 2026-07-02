from __future__ import annotations

from pathlib import Path

from .base import LLMClient
from .mock import MockLLMClient
from .transformers_client import TransformersLLMClient
from .vllm_client import VLLMClient


def build_llm_client(
    backend: str,
    *,
    model_path: str | Path = "models/Qwen3-4B",
    device_map: str = "auto",
    torch_dtype: str = "auto",
    max_model_len: int | None = 8192,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    enforce_eager: bool = False,
    speculative_model_path: str | Path | None = None,
    speculative_tokens: int = 3,
) -> LLMClient:
    if backend == "mock":
        return MockLLMClient()
    if backend == "transformers":
        return TransformersLLMClient(model_path, device_map=device_map, torch_dtype=torch_dtype)
    if backend == "vllm":
        return VLLMClient(
            model_path,
            dtype=torch_dtype,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            speculative_model_path=speculative_model_path,
            speculative_tokens=speculative_tokens,
        )
    raise ValueError(f"Unsupported backend: {backend}")
