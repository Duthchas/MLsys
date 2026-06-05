from __future__ import annotations

import math
import torch
from einops import einsum

from cs336_basics.softmax import softmax


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Computes the scaled dot-product attention.

    Args:
        Q: Query tensor of shape (batch_size, ..., seq_len, d_k)
        K: Key tensor of shape (batch_size, ..., seq_len, d_k)
        V: Value tensor of shape (batch_size, ..., seq_len, d_v)
        mask: Optional boolean mask of shape (seq_len, seq_len) or (..., seq_len, seq_len).
              True values are attended to, False values are masked out.

    Returns:
        Output tensor of shape (batch_size, ..., seq_len, d_v)
    """
    d_k = Q.size(-1)

    scores = einsum(Q, K, "... q d_k, ... k d_k -> ... q k")
    scores = scores / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    attention_probs = softmax(scores, dim=-1)
    output = einsum(attention_probs, V, "... q k, ... k d_v -> ... q d_v")

    return output
