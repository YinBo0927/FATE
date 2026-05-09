from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from fate.training.pfpo import clipped_pfpo_loss, group_relative_advantages
from fate.utils.io import read_jsonl


@dataclass
class PFPOCompletion:
    text: str
    reward: float


@dataclass
class PFPOGroup:
    prompt_id: str
    prompt: str
    completions: list[PFPOCompletion]

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "PFPOGroup":
        return cls(
            prompt_id=str(row.get("prompt_id") or row.get("failure_id") or row.get("id")),
            prompt=str(row["prompt"]),
            completions=[
                PFPOCompletion(text=str(item["text"]), reward=float(item["reward"]))
                for item in row["completions"]
            ],
        )


class PFPOGroupDataset(Dataset):
    def __init__(self, groups: list[PFPOGroup], tokenizer: Any, max_length: int):
        self.groups = groups
        self.tokenizer = tokenizer
        self.max_length = max_length

    @classmethod
    def from_jsonl(cls, path: str | Path, tokenizer: Any, max_length: int) -> "PFPOGroupDataset":
        groups = [PFPOGroup.from_mapping(row) for row in read_jsonl(path)]
        return cls(groups=groups, tokenizer=tokenizer, max_length=max_length)

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index: int) -> dict[str, Any]:
        group = self.groups[index]
        encoded_group = []
        rewards = []
        for completion in group.completions:
            prompt_ids = self.tokenizer(group.prompt, add_special_tokens=False)["input_ids"]
            completion_ids = self.tokenizer(completion.text, add_special_tokens=False)["input_ids"]
            input_ids = (prompt_ids + completion_ids + [self.tokenizer.eos_token_id])[-self.max_length :]
            target_start = max(0, len(input_ids) - len(completion_ids) - 1)
            completion_mask = [0] * target_start + [1] * (len(input_ids) - target_start)
            encoded_group.append(
                {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
                    "completion_mask": torch.tensor(completion_mask, dtype=torch.float32),
                }
            )
            rewards.append(completion.reward)
        return {"group": encoded_group, "rewards": torch.tensor(rewards, dtype=torch.float32)}


class PFPOCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch_size = len(batch)
        group_size = max(len(item["group"]) for item in batch)
        max_len = max(seq["input_ids"].numel() for item in batch for seq in item["group"])
        input_ids = torch.full((batch_size, group_size, max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, group_size, max_len), dtype=torch.long)
        completion_mask = torch.zeros((batch_size, group_size, max_len), dtype=torch.float32)
        rewards = torch.zeros((batch_size, group_size), dtype=torch.float32)
        group_mask = torch.zeros((batch_size, group_size), dtype=torch.float32)

        for batch_index, item in enumerate(batch):
            for group_index, seq in enumerate(item["group"]):
                length = seq["input_ids"].numel()
                input_ids[batch_index, group_index, :length] = seq["input_ids"]
                attention_mask[batch_index, group_index, :length] = seq["attention_mask"]
                completion_mask[batch_index, group_index, :length] = seq["completion_mask"]
                rewards[batch_index, group_index] = item["rewards"][group_index]
                group_mask[batch_index, group_index] = 1.0

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "completion_mask": completion_mask,
            "rewards": rewards,
            "group_mask": group_mask,
        }


def train_offline_pfpo(
    model: Any,
    old_model: Any,
    ref_model: Any,
    tokenizer: Any,
    dataset: PFPOGroupDataset,
    output_dir: str | Path,
    *,
    learning_rate: float,
    clip_epsilon: float,
    kl_beta: float,
    batch_size: int,
    gradient_accumulation_steps: int,
    max_steps: int,
) -> None:
    device = next(model.parameters()).device
    old_model.eval()
    ref_model.eval()
    for frozen in old_model.parameters():
        frozen.requires_grad_(False)
    for frozen in ref_model.parameters():
        frozen.requires_grad_(False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=PFPOCollator(tokenizer.pad_token_id),
    )

    step = 0
    optimizer.zero_grad(set_to_none=True)
    while step < max_steps:
        for batch in tqdm(loader, desc="pfpo"):
            batch = {key: value.to(device) for key, value in batch.items()}
            batch_size_now, group_size, seq_len = batch["input_ids"].shape
            flat = {
                "input_ids": batch["input_ids"].reshape(batch_size_now * group_size, seq_len),
                "attention_mask": batch["attention_mask"].reshape(batch_size_now * group_size, seq_len),
            }
            completion_mask = batch["completion_mask"].reshape(batch_size_now, group_size, seq_len)

            new_logprobs = _token_logprobs(model, **flat).reshape(batch_size_now, group_size, -1)
            with torch.no_grad():
                old_logprobs = _token_logprobs(old_model, **flat).reshape(batch_size_now, group_size, -1)
                ref_logprobs = _token_logprobs(ref_model, **flat).reshape(batch_size_now, group_size, -1)

            shifted_mask = completion_mask[:, :, 1:]
            advantages = group_relative_advantages(batch["rewards"])
            loss = clipped_pfpo_loss(
                new_logprobs=new_logprobs,
                old_logprobs=old_logprobs,
                ref_logprobs=ref_logprobs,
                advantages=advantages,
                completion_mask=shifted_mask,
                clip_epsilon=clip_epsilon,
                kl_beta=kl_beta,
            )
            (loss / gradient_accumulation_steps).backward()
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            step += 1
            if step >= max_steps:
                break

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def _token_logprobs(model: Any, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :]
    labels = input_ids[:, 1:]
    return torch.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)

