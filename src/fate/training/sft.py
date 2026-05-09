from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments


@dataclass
class CompletionOnlyExample:
    prompt: str
    target: str


class CompletionOnlyDataset(Dataset):
    def __init__(self, examples: list[CompletionOnlyExample], tokenizer: Any, max_length: int):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        example = self.examples[index]
        prompt_ids = self.tokenizer(example.prompt, add_special_tokens=False)["input_ids"]
        target_ids = self.tokenizer(example.target, add_special_tokens=False)["input_ids"]
        input_ids = (prompt_ids + target_ids + [self.tokenizer.eos_token_id])[-self.max_length :]
        prompt_cut = max(0, len(input_ids) - len(target_ids) - 1)
        labels = [-100] * prompt_cut + input_ids[prompt_cut:]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
        }


class CompletionCollator:
    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer

    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        max_len = max(item["input_ids"].shape[0] for item in batch)
        padded = {"input_ids": [], "labels": [], "attention_mask": []}
        for item in batch:
            pad = max_len - item["input_ids"].shape[0]
            padded["input_ids"].append(
                torch.nn.functional.pad(
                    item["input_ids"], (0, pad), value=self.tokenizer.pad_token_id
                )
            )
            padded["labels"].append(torch.nn.functional.pad(item["labels"], (0, pad), value=-100))
            padded["attention_mask"].append(
                torch.nn.functional.pad(item["attention_mask"], (0, pad), value=0)
            )
        return {key: torch.stack(value) for key, value in padded.items()}


def make_trainer(
    model: Any,
    tokenizer: Any,
    examples: list[CompletionOnlyExample],
    output_dir: str,
    *,
    max_length: int,
    training_args: dict[str, Any],
) -> Trainer:
    args = TrainingArguments(output_dir=output_dir, **training_args)
    dataset = CompletionOnlyDataset(examples, tokenizer=tokenizer, max_length=max_length)
    return Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=CompletionCollator(tokenizer),
    )

