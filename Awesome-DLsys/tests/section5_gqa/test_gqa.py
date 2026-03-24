import os
import sys

# Use numpy backend for determinism; skip data imports.
os.environ.setdefault("NEEDLE_BACKEND", "np")
os.environ.setdefault("NEEDLE_DISABLE_DATA", "1")

sys.path.append("./python")

import numpy as np
import pytest
import torch

import types

# Stub out needle.data to avoid importing datasets when using numpy backend.
sys.modules.setdefault("needle.data", types.ModuleType("needle.data"))

import needle as ndl
import needle.nn as nn
from needle.nn.nn_basic import Parameter
from needle.nn.nn_transformer import apply_rotary_pos_emb

try:
    import transformers  # noqa: F401
    _HAS_TRANSFORMERS = True
except Exception:
    _HAS_TRANSFORMERS = False


_DEVICES = [ndl.cpu()]
_DEVICE_IDS = ["cpu"]
if hasattr(ndl, "cuda"):
    _cuda = ndl.cuda()
    _DEVICES.append(
        pytest.param(
            _cuda,
            marks=pytest.mark.skipif(
                not _cuda.enabled(), reason="CUDA device is not available"
            ),
        )
    )
    _DEVICE_IDS.append("cuda")


def _apply_rope_np(x, base=10000.0):
    """HF-style RoPE for numpy arrays."""
    b, s, h, d = x.shape
    half = d // 2
    inv = base ** (-np.arange(0, half, dtype=np.float32) / half)
    freqs = np.outer(np.arange(s, dtype=np.float32), inv)
    cos = np.concatenate([np.cos(freqs), np.cos(freqs)], axis=-1)[None, :, None, :]
    sin = np.concatenate([np.sin(freqs), np.sin(freqs)], axis=-1)[None, :, None, :]
    x1 = x[..., :half]
    x2 = x[..., half:]
    rotated = np.concatenate([-x2, x1], axis=-1)
    return x * cos + rotated * sin


@pytest.mark.parametrize("device", _DEVICES, ids=_DEVICE_IDS)
def test_gqa_head_sharing_matches_numpy(device):
    base = 10000.0
    batch, seq_len, dim = 1, 2, 2
    num_head, num_kv_heads, dim_head = 2, 1, 2
    x_np = np.array([[[0.2, -0.3], [0.4, 0.1]]], dtype=np.float32)

    layer = nn.AttentionLayer(
        q_features=dim,
        num_head=num_head,
        dim_head=dim_head,
        num_kv_heads=num_kv_heads,
        dropout=0.0,
        causal=False,
        device=device,
        dtype="float32",
        rope_base=base,
    )

    # Remove prenorm effects.
    layer.prenorm_q = nn.Identity()
    layer.prenorm_k = nn.Identity()
    layer.prenorm_v = nn.Identity()

    # Weights chosen so q heads duplicate input; single kv head is identity; output sums both heads.
    wq = np.array([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=np.float32)  # (2,4)
    wk = np.eye(dim, dtype=np.float32)  # (2,2)
    wv = np.eye(dim, dtype=np.float32)  # (2,2)
    wout = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=np.float32)  # (4,2)

    layer.q_projection.weight = Parameter(ndl.Tensor(wq, device=device))
    layer.k_projection.weight = Parameter(ndl.Tensor(wk, device=device))
    layer.v_projection.weight = Parameter(ndl.Tensor(wv, device=device))
    layer.out_projection.weight = Parameter(ndl.Tensor(wout, device=device))

    out = layer(ndl.Tensor(x_np, device=device)).numpy()

    # Manual GQA attention with shared K/V.
    q_proj = x_np @ wq  # (1,2,4)
    q_proj = q_proj.reshape(batch, seq_len, num_head, dim_head)
    k_proj = x_np @ wk  # (1,2,2) -> (1,2,1,2)
    k_proj = k_proj.reshape(batch, seq_len, num_kv_heads, dim_head)
    v_proj = k_proj.copy()

    q_rot = _apply_rope_np(q_proj, base=base)
    k_rot = _apply_rope_np(k_proj, base=base)

    # Repeat kv heads to match query heads.
    repeat = num_head // num_kv_heads
    k_repeat = np.repeat(k_rot, repeat, axis=2)
    v_repeat = np.repeat(v_proj, repeat, axis=2)

    q = np.transpose(q_rot, (0, 2, 1, 3))
    k = np.transpose(k_repeat, (0, 2, 1, 3))
    v = np.transpose(v_repeat, (0, 2, 1, 3))

    logits = np.matmul(q, np.transpose(k, (0, 1, 3, 2))) / np.sqrt(dim_head)
    logits = logits - logits.max(axis=-1, keepdims=True)
    probs = np.exp(logits)
    probs = probs / probs.sum(axis=-1, keepdims=True)

    attn_out = np.matmul(probs, v)
    attn_out = np.transpose(attn_out, (0, 2, 1, 3)).reshape(batch, seq_len, num_head * dim_head)
    expected = attn_out @ wout

    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("device", _DEVICES, ids=_DEVICE_IDS)
@pytest.mark.skipif(not _HAS_TRANSFORMERS, reason="transformers not installed")
def test_gqa_matches_hf_llama_attention(device):
    from transformers.models.llama.modeling_llama import LlamaConfig, LlamaAttention

    base = 10000.0
    batch, seq_len, hidden_size = 2, 3, 6
    num_head, num_kv_heads, dim_head = 3, 1, 2
    np.random.seed(42)
    x_np = np.random.randn(batch, seq_len, hidden_size).astype(np.float32)

    layer = nn.AttentionLayer(
        q_features=hidden_size,
        num_head=num_head,
        dim_head=dim_head,
        num_kv_heads=num_kv_heads,
        dropout=0.0,
        causal=False,
        device=device,
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

    layer.q_projection.weight = Parameter(ndl.Tensor(wq, device=device))
    layer.k_projection.weight = Parameter(ndl.Tensor(wk, device=device))
    layer.v_projection.weight = Parameter(ndl.Tensor(wv, device=device))
    layer.out_projection.weight = Parameter(ndl.Tensor(wo, device=device))

    config = LlamaConfig(
        hidden_size=hidden_size,
        num_attention_heads=num_head,
        num_key_value_heads=num_kv_heads,
        rope_theta=base,
        max_position_embeddings=seq_len,
        rms_norm_eps=1e-5,
        attention_bias=False,
    )
    # Ensure explicit attn implementation for HF versions that require it.
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
        # Fallback for HF versions requiring explicit rotary embeddings.
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

        rope = LlamaRotaryEmbedding(config)
        try:
            cos, sin = rope(torch_x, position_ids=position_ids)
        except TypeError:
            cos, sin = rope(torch_x, seq_len=seq_len)

        try:
            hf_out = hf_attn(
                torch_x,
                attention_mask=None,
                position_embeddings=(cos, sin),
                output_attentions=False,
                use_cache=False,
            )[0]
        except TypeError:
            hf_out = hf_attn(
                torch_x,
                None,
                (cos, sin),
                None,
                False,
                False,
            )[0]
    hf_out = hf_out.detach().cpu().numpy()

    ours = layer(ndl.Tensor(x_np, device=device)).numpy()
    np.testing.assert_allclose(ours, hf_out, rtol=1e-5, atol=1e-6)
