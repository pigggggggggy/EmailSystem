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
    max_num_batched_tokens: int | None = None,
    max_num_seqs: int | None = None,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    enforce_eager: bool = False,
    enable_dbo: bool = False,
    quantization: str | None = None,
    speculative_model_path: str | Path | None = None,
    speculative_tokens: int = 3,
    ngram_prompt_lookup_min: int | None = None,
    ngram_prompt_lookup_max: int | None = None,
    collect_speculative_metrics: bool = False,
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
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            enable_dbo=enable_dbo,
            quantization=quantization,
            speculative_model_path=speculative_model_path,
            speculative_tokens=speculative_tokens,
            ngram_prompt_lookup_min=ngram_prompt_lookup_min,
            ngram_prompt_lookup_max=ngram_prompt_lookup_max,
            collect_speculative_metrics=collect_speculative_metrics,
        )
    raise ValueError(f"Unsupported backend: {backend}")
