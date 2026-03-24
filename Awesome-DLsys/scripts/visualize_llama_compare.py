#!/usr/bin/env python3
"""
Visualize module-by-module parity between our Needle implementation and HuggingFace LLaMA.

Compares RoPE, SwiGLU MLP, and GQA attention outputs on the same random inputs/weights
and plots heatmaps of absolute differences.
"""
import os
import sys
import types
from typing import Tuple

os.environ.setdefault("NEEDLE_BACKEND", "np")
os.environ.setdefault("NEEDLE_DISABLE_DATA", "1")

# Avoid dataset imports in needle.data when using numpy backend.
sys.modules.setdefault("needle.data", types.ModuleType("needle.data"))
sys.path.append("./python")

import numpy as np
import torch

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - visualization only
    raise SystemExit(f"matplotlib is required to run this script: {exc}")

try:
    from transformers.models.llama.modeling_llama import (
        LlamaAttention,
        LlamaConfig,
        LlamaMLP,
        LlamaRMSNorm,
        LlamaRotaryEmbedding,
        apply_rotary_pos_emb as hf_apply_rotary_pos_emb,
    )
except Exception as exc:  # pragma: no cover - visualization only
    raise SystemExit(f"transformers with LLaMA support is required: {exc}")

import needle as ndl
import needle.nn as nn
from needle.nn.nn_basic import Parameter
from needle.nn.nn_transformer import apply_rotary_pos_emb


def plot_heatmap(ax, data: np.ndarray, title: str):
    im = ax.imshow(data, aspect="auto", cmap="magma")
    ax.set_title(title)
    ax.set_xlabel("Features")
    ax.set_ylabel("Tokens")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def compare_rope(ax):
    base = 10000.0
    batch, seq_len, num_head, dim_head = 2, 6, 4, 8
    x_np = np.random.randn(batch, seq_len, num_head, dim_head).astype(np.float32)

    ours = apply_rotary_pos_emb(ndl.Tensor(x_np), base=base).numpy()

    x_t = torch.tensor(x_np)
    rope = LlamaRotaryEmbedding(
        LlamaConfig(
            hidden_size=num_head * dim_head,
            num_attention_heads=num_head,
            max_position_embeddings=seq_len,
            rope_theta=base,
        )
    )
    try:
        cos, sin = rope(x_t, position_ids=torch.arange(seq_len)[None, :])
    except TypeError:
        cos, sin = rope(x_t, seq_len=seq_len)
    hf_out = hf_apply_rotary_pos_emb(x_t, x_t, cos, sin, None, unsqueeze_dim=2)[0].detach().cpu().numpy()

    diff = np.abs(ours - hf_out).reshape(batch * seq_len, num_head * dim_head)
    plot_heatmap(ax, diff, "RoPE |abs diff| (ours vs HF)")


def run_swiglu_needle(layer: nn.TransformerLayer, x_np: np.ndarray) -> np.ndarray:
    b, s, d = x_np.shape
    x = ndl.Tensor(x_np, device=ndl.cpu())
    norm_in = ndl.ops.reshape(x, (b * s, d))
    normed = layer.ff_norm(norm_in)

    ff1 = layer.ff_linear1(normed)
    ff1 = ndl.ops.reshape(ff1, (b, s, 2, layer.hidden_size))
    u, v = ndl.ops.split(ff1, axis=2)
    u = ndl.ops.reshape(u, (b, s, layer.hidden_size))
    v = ndl.ops.reshape(v, (b, s, layer.hidden_size))

    exp_neg_v = ndl.ops.exp(ndl.ops.negate(v))
    denom = ndl.ops.add_scalar(exp_neg_v, 1.0)
    sigmoid_v = ndl.ops.power_scalar(denom, -1)
    swish_v = v * sigmoid_v

    hidden = u * swish_v
    hidden = ndl.ops.reshape(hidden, (b * s, layer.hidden_size))
    hidden = layer.ff_dropout(hidden)
    hidden = layer.ff_linear2(hidden)
    hidden = layer.ff_output_dropout(hidden)
    return ndl.ops.reshape(hidden, (b, s, d)).numpy()


def compare_swiglu(ax):
    d_model, hidden = 12, 16
    batch, seq = 2, 4
    np.random.seed(7)
    x_np = np.random.randn(batch, seq, d_model).astype(np.float32)

    layer = nn.TransformerLayer(
        d_model,
        num_head=3,
        dim_head=4,
        hidden_size=hidden,
        dropout=0.0,
        causal=False,
        device=ndl.cpu(),
    )
    # Remove attention influence by zeroing projections.
    zeros_attn_w = np.zeros_like(layer.attention.q_projection.weight.numpy())
    layer.attention.q_projection.weight = Parameter(ndl.Tensor(zeros_attn_w))
    layer.attention.k_projection.weight = Parameter(
        ndl.Tensor(np.zeros_like(layer.attention.k_projection.weight.numpy()))
    )
    layer.attention.v_projection.weight = Parameter(
        ndl.Tensor(np.zeros_like(layer.attention.v_projection.weight.numpy()))
    )
    layer.attention.out_projection.weight = Parameter(
        ndl.Tensor(np.zeros_like(layer.attention.out_projection.weight.numpy()))
    )

    # Shared weights for MLP vs HF.
    w1 = np.random.randn(d_model, 2 * hidden).astype(np.float32)
    w2 = np.random.randn(hidden, d_model).astype(np.float32)
    layer.ff_linear1.weight = Parameter(ndl.Tensor(w1))
    layer.ff_linear1.bias = Parameter(ndl.Tensor(np.zeros((1, 2 * hidden), dtype=np.float32)))
    layer.ff_linear2.weight = Parameter(ndl.Tensor(w2))
    layer.ff_linear2.bias = Parameter(ndl.Tensor(np.zeros((1, d_model), dtype=np.float32)))
    layer.ff_norm.weight = Parameter(ndl.Tensor(np.ones((d_model,), dtype=np.float32)))

    config = LlamaConfig(
        hidden_size=d_model,
        intermediate_size=hidden,
        num_attention_heads=1,
        rms_norm_eps=1e-5,
    )
    hf_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    hf_mlp = LlamaMLP(config)
    with torch.no_grad():
        hf_mlp.up_proj.weight.copy_(torch.tensor(w1[:, :hidden].T))
        hf_mlp.gate_proj.weight.copy_(torch.tensor(w1[:, hidden:].T))
        hf_mlp.down_proj.weight.copy_(torch.tensor(w2.T))
        if hf_mlp.up_proj.bias is not None:
            hf_mlp.up_proj.bias.zero_()
        if hf_mlp.gate_proj.bias is not None:
            hf_mlp.gate_proj.bias.zero_()
        if hf_mlp.down_proj.bias is not None:
            hf_mlp.down_proj.bias.zero_()
        hf_norm.weight.copy_(torch.tensor(layer.ff_norm.weight.numpy()))

    ours = run_swiglu_needle(layer, x_np)
    hf_out = hf_mlp(hf_norm(torch.tensor(x_np))).detach().cpu().numpy()

    diff = np.abs(ours - hf_out).reshape(batch * seq, d_model)
    plot_heatmap(ax, diff, "SwiGLU MLP |abs diff| (ours vs HF)")


def compare_gqa(ax):
    base = 10000.0
    batch, seq_len, hidden_size = 2, 4, 6
    num_head, num_kv_heads, dim_head = 3, 1, 2
    np.random.seed(13)
    x_np = np.random.randn(batch, seq_len, hidden_size).astype(np.float32)

    layer = nn.AttentionLayer(
        q_features=hidden_size,
        num_head=num_head,
        dim_head=dim_head,
        num_kv_heads=num_kv_heads,
        dropout=0.0,
        causal=False,
        device=ndl.cpu(),
        dtype="float32",
        rope_base=base,
    )
    # Remove prenorm effects.
    layer.prenorm_q = nn.Identity()
    layer.prenorm_k = nn.Identity()
    layer.prenorm_v = nn.Identity()

    wq = np.random.randn(hidden_size, num_head * dim_head).astype(np.float32)
    wk = np.random.randn(hidden_size, num_kv_heads * dim_head).astype(np.float32)
    wv = np.random.randn(hidden_size, num_kv_heads * dim_head).astype(np.float32)
    wo = np.random.randn(num_head * dim_head, hidden_size).astype(np.float32)

    layer.q_projection.weight = Parameter(ndl.Tensor(wq))
    layer.k_projection.weight = Parameter(ndl.Tensor(wk))
    layer.v_projection.weight = Parameter(ndl.Tensor(wv))
    layer.out_projection.weight = Parameter(ndl.Tensor(wo))

    config = LlamaConfig(
        hidden_size=hidden_size,
        num_attention_heads=num_head,
        num_key_value_heads=num_kv_heads,
        rope_theta=base,
        max_position_embeddings=seq_len,
        rms_norm_eps=1e-5,
        attention_bias=False,
    )
    setattr(config, "_attn_implementation", "eager")
    hf_attn = LlamaAttention(config, layer_idx=0)
    with torch.no_grad():
        hf_attn.q_proj.weight.copy_(torch.tensor(wq.T))
        hf_attn.k_proj.weight.copy_(torch.tensor(wk.T))
        hf_attn.v_proj.weight.copy_(torch.tensor(wv.T))
        hf_attn.o_proj.weight.copy_(torch.tensor(wo.T))

    torch_x = torch.tensor(x_np)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).repeat(batch, 1)
    try:
        hf_out = hf_attn(
            torch_x,
            attention_mask=None,
            position_ids=position_ids,
            output_attentions=False,
            use_cache=False,
        )[0]
    except (TypeError, AttributeError):
        rope = LlamaRotaryEmbedding(config)
        try:
            cos, sin = rope(torch_x, position_ids=position_ids)
        except TypeError:
            cos, sin = rope(torch_x, seq_len=seq_len)
        hf_out = hf_attn(
            torch_x,
            attention_mask=None,
            position_embeddings=(cos, sin),
            output_attentions=False,
            use_cache=False,
        )[0]
    hf_out = hf_out.detach().cpu().numpy()

    ours = layer(ndl.Tensor(x_np)).numpy()
    diff = np.abs(ours - hf_out).reshape(batch * seq_len, hidden_size)
    plot_heatmap(ax, diff, "GQA Attention |abs diff| (ours vs HF)")


def main():
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    compare_rope(axs[0])
    compare_swiglu(axs[1])
    compare_gqa(axs[2])
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
