from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class Generation:
    text: str
    logprob: float = 0.0


class PolicyClient(Protocol):
    def generate(
        self,
        prompts: list[str],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        num_return_sequences: int = 1,
    ) -> list[list[Generation]]:
        """Generate one or more completions for each prompt."""

