from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .base import GenerationResult
from .chat_prompts import messages_for_task


class TransformersLLMClient:
    """Local Hugging Face transformers client for Qwen-style chat models."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        trust_remote_code: bool = True,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "The transformers backend requires torch and transformers. "
                "Install them with: pip install -e '.[qwen]'"
            ) from exc

        self.model_path = str(model_path)
        dtype = self._resolve_dtype(torch, torch_dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=trust_remote_code)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )

    def generate(self, prompt: str, *, task: str, max_tokens: int = 512) -> GenerationResult:
        start = time.perf_counter()
        messages = messages_for_task(prompt, task)
        input_ids = self._encode_messages(messages)
        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generated_ids = output_ids[0][input_ids.shape[-1] :]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        latency_ms = (time.perf_counter() - start) * 1000
        return GenerationResult(
            text=text,
            input_tokens=int(input_ids.shape[-1]),
            output_tokens=int(generated_ids.shape[-1]),
            latency_ms=latency_ms,
        )

    def _encode_messages(self, messages: list[dict[str, str]]) -> Any:
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    enable_thinking=False,
                ).to(self.model.device)
            except TypeError:
                return self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                ).to(self.model.device)
        text = "\n".join(f"{item['role']}: {item['content']}" for item in messages) + "\nassistant:"
        return self.tokenizer(text, return_tensors="pt").input_ids.to(self.model.device)

    def _resolve_dtype(self, torch: Any, torch_dtype: str) -> Any:
        if torch_dtype == "auto":
            return "auto"
        if torch_dtype == "float16":
            return torch.float16
        if torch_dtype == "bfloat16":
            return torch.bfloat16
        if torch_dtype == "float32":
            return torch.float32
        raise ValueError(f"Unsupported torch dtype: {torch_dtype}")
