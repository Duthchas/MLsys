from __future__ import annotations

import torch


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    shifted = x - torch.max(x, dim=dim, keepdim=True).values
    exp_shifted = torch.exp(shifted)
    return exp_shifted / torch.sum(exp_shifted, dim=dim, keepdim=True)


def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    shifted = inputs - torch.max(inputs, dim=-1, keepdim=True).values
    log_sum_exp = torch.log(torch.sum(torch.exp(shifted), dim=-1))
    target_logits = torch.gather(
        shifted,
        dim=-1,
        index=targets.long().unsqueeze(-1),
    ).squeeze(-1)
    return (log_sum_exp - target_logits).mean()
