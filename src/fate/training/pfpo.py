from __future__ import annotations

import torch


def group_relative_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """Compute A_i = r_i - mean_j r_j for grouped PFPO completions.

    `rewards` shape: [batch, group].
    """

    return rewards - rewards.mean(dim=1, keepdim=True)


def clipped_pfpo_loss(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor,
    *,
    clip_epsilon: float,
    kl_beta: float,
) -> torch.Tensor:
    """Sequence-token implementation of the paper's PFPO objective.

    Shapes:
    - logprob tensors: [batch, group, seq]
    - advantages: [batch, group]
    - completion_mask: [batch, group, seq]
    """

    ratio = torch.exp(new_logprobs - old_logprobs)
    adv = advantages.unsqueeze(-1)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    policy_term = torch.minimum(ratio * adv, clipped_ratio * adv)
    kl_term = new_logprobs - ref_logprobs
    masked = (policy_term - kl_beta * kl_term) * completion_mask
    denom = completion_mask.sum().clamp_min(1.0)
    return -masked.sum() / denom

