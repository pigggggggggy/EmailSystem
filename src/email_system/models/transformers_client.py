from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .base import GenerationResult


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
        messages = self._messages_for_task(prompt, task)
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
            return self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(self.model.device)
        text = "\n".join(f"{item['role']}: {item['content']}" for item in messages) + "\nassistant:"
        return self.tokenizer(text, return_tensors="pt").input_ids.to(self.model.device)

    def _messages_for_task(self, prompt: str, task: str) -> list[dict[str, str]]:
        system = "你是企业邮件处理助手。输出必须严格遵守用户要求，不要添加无关解释。"
        if task == "classify_email":
            user = (
                "请分析下面邮件，只输出 JSON："
                "{\"category\": \"invoice|support|meeting|sales|spam|personal|other\", "
                "\"priority\": \"low|normal|high|urgent\", \"confidence\": 0.0}\n\n"
                f"{prompt}"
            )
        elif task == "summarize_email":
            user = f"请用一句话总结下面邮件，只输出 JSON：{{\"summary\": \"...\", \"confidence\": 0.0}}\n\n{prompt}"
        elif task == "extract_action_items":
            user = (
                "请提取下面邮件里的待办事项，只输出 JSON："
                "{\"action_items\": [{\"owner\": null, \"task\": \"...\", \"due\": null}]}\n\n"
                f"{prompt}"
            )
        elif task == "draft_reply":
            user = f"请根据下面邮件生成一段简洁、专业的中文回复草稿。只输出回复正文。\n\n{prompt}"
        else:
            user = prompt
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

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
