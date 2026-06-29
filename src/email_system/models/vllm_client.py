from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .base import GenerationResult
from .chat_prompts import messages_for_task


class VLLMClient:
    """Local vLLM backend for Qwen-style chat models."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        dtype: str = "auto",
        max_model_len: int | None = 8192,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        trust_remote_code: bool = True,
        enforce_eager: bool = False,
    ) -> None:
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise RuntimeError("The vLLM backend requires vllm. Install it with: pip install -e '.[vllm]'") from exc

        self.model_path = str(model_path)
        self.sampling_params_cls = SamplingParams
        kwargs: dict[str, Any] = {
            "model": self.model_path,
            "tokenizer": self.model_path,
            "dtype": dtype,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "trust_remote_code": trust_remote_code,
            "enforce_eager": enforce_eager,
        }
        if max_model_len is not None:
            kwargs["max_model_len"] = max_model_len
        self.llm = LLM(**kwargs)
        self.tokenizer = self.llm.get_tokenizer()

    def generate(self, prompt: str, *, task: str, max_tokens: int = 512) -> GenerationResult:
        start = time.perf_counter()
        rendered_prompt = self._render_prompt(messages_for_task(prompt, task))
        sampling_params = self.sampling_params_cls(temperature=0.0, max_tokens=max_tokens)
        outputs = self.llm.generate([rendered_prompt], sampling_params, use_tqdm=False)
        request_output = outputs[0]
        completion = request_output.outputs[0]
        text = completion.text.strip()
        latency_ms = (time.perf_counter() - start) * 1000
        output_tokens = len(getattr(completion, "token_ids", []) or [])
        input_tokens = len(getattr(request_output, "prompt_token_ids", []) or [])
        return GenerationResult(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
        )

    def _render_prompt(self, messages: list[dict[str, str]]) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    enable_thinking=False,
                )
            except TypeError:
                return self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )
        return "\n".join(f"{item['role']}: {item['content']}" for item in messages) + "\nassistant:"
