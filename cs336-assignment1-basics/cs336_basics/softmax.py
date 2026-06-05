from __future__ import annotations

import torch


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    shifted = x - torch.max(x, dim=dim, keepdim=True).values
    exp_shifted = torch.exp(shifted)
    return exp_shifted / torch.sum(exp_shifted, dim=dim, keepdim=True)
