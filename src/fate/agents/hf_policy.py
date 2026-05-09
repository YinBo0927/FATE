from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from fate.agents.base import Generation


@dataclass
class HFPolicyClient:
    model: Any
    tokenizer: Any
    device: str = "cuda"

    @classmethod
    def from_pretrained(
        cls,
        name_or_path: str,
        *,
        torch_dtype: str = "bfloat16",
        trust_remote_code: bool = True,
        attn_implementation: str | None = None,
    ) -> "HFPolicyClient":
        dtype = getattr(torch, torch_dtype)
        kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "device_map": "auto",
            "trust_remote_code": trust_remote_code,
        }
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        tokenizer = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=trust_remote_code)
        model = AutoModelForCausalLM.from_pretrained(name_or_path, **kwargs)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        return cls(model=model, tokenizer=tokenizer)

    @torch.inference_mode()
    def generate(
        self,
        prompts: list[str],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        num_return_sequences: int = 1,
    ) -> list[list[Generation]]:
        encoded = self.tokenizer(prompts, padding=True, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
            return_dict_in_generate=True,
            output_scores=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        prompt_len = encoded["input_ids"].shape[1]
        decoded = self.tokenizer.batch_decode(
            outputs.sequences[:, prompt_len:], skip_special_tokens=True
        )
        grouped: list[list[Generation]] = []
        index = 0
        for _ in prompts:
            group = []
            for _ in range(num_return_sequences):
                group.append(Generation(text=decoded[index].strip(), logprob=0.0))
                index += 1
            grouped.append(group)
        return grouped

