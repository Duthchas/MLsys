from __future__ import annotations

import math

import torch
from einops import einsum, rearrange, pack, unpack, repeat
from torch import nn

from cs336_basics.attention import scaled_dot_product_attention

class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )

        std = math.sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.weight = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        )

        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x_float = x.to(torch.float32)
        inv_rms = torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x_float * inv_rms * self.weight).to(in_dtype)


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        if d_ff is None:
            d_ff = max(64, int(8 * d_model / 3) // 64 * 64)

        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.w1(x)
        return self.w2(gate * torch.sigmoid(gate) * self.w3(x))


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()

        positions = torch.arange(max_seq_len, device=device)
        dim_indices = torch.arange(0, d_k, 2, device=device)
        inv_freq = theta ** (-dim_indices / d_k)
        angles = torch.outer(positions, inv_freq)

        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        x_pairs = rearrange(x, "... seq (pair two) -> ... seq pair two", two=2)
        x_even, x_odd = x_pairs.unbind(dim=-1)

        cos = self.cos[token_positions].to(dtype=x.dtype, device=x.device)
        sin = self.sin[token_positions].to(dtype=x.dtype, device=x.device)

        rotated = torch.stack(
            (
                x_even * cos - x_odd * sin,
                x_even * sin + x_odd * cos,
            ),
            dim=-1,
        )

        return rearrange(rotated, "... seq pair two -> ... seq (pair two)")


class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope_theta: float | None = None,
        rope_max_seq_len: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        if rope_theta is not None and rope_max_seq_len is not None:
            self.rope = RotaryPositionalEmbedding(
                theta=rope_theta,
                d_k=self.d_k,
                max_seq_len=rope_max_seq_len,
                device=device,
            )
        else:
            self.rope = None

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        Q = rearrange(q, "... seq (h d_k) -> ... h seq d_k", h=self.num_heads)
        K = rearrange(k, "... seq (h d_k) -> ... h seq d_k", h=self.num_heads)
        V = rearrange(v, "... seq (h d_v) -> ... h seq d_v", h=self.num_heads)

        if self.rope is not None:
            if token_positions is None:
                seq_len = x.shape[-2]
                token_positions = torch.arange(seq_len, device=x.device)
                
            token_positions = token_positions.expand(*x.shape[:-1])
            token_positions_expanded = repeat(token_positions, "... seq -> ... h seq", h=self.num_heads)

            Q_flat, ps = pack([Q], "* seq d_k")
            K_flat, _ = pack([K], "* seq d_k")
            token_positions_flat, _ = pack([token_positions_expanded], "* seq")

            Q_rope = self.rope(Q_flat, token_positions_flat)
            K_rope = self.rope(K_flat, token_positions_flat)

            [Q] = unpack(Q_rope, ps, "* seq d_k")
            [K] = unpack(K_rope, ps, "* seq d_k")

        seq_len = Q.shape[-2]
        mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=x.device))

        attn_out = scaled_dot_product_attention(Q, K, V, mask=mask)

        out = rearrange(attn_out, "... h seq d_v -> ... seq (h d_v)")

        return self.output_proj(out)
