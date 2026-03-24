import os
import sys

# Prefer numpy backend and disable data import for portability.
os.environ.setdefault("NEEDLE_BACKEND", "np")
os.environ.setdefault("NEEDLE_DISABLE_DATA", "1")

sys.path.append("./python")

import numpy as np
import pytest

import needle as ndl
import needle.nn as nn
from needle.nn.nn_transformer import apply_rotary_pos_emb


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

try:
    import torch
    from transformers.models.llama.modeling_llama import (
        LlamaRotaryEmbedding,
        apply_rotary_pos_emb as hf_apply_rotary_pos_emb,
    )

    _HAS_TRANSFORMERS = True
except Exception:
    _HAS_TRANSFORMERS = False


@pytest.mark.parametrize("device", _DEVICES, ids=_DEVICE_IDS)
@pytest.mark.skipif(not _HAS_TRANSFORMERS, reason="transformers not installed")
def test_rope_matches_hf_llama_rotary(device):
    base = 10000.0
    batch, seq_len, num_head, dim_head = 2, 4, 3, 8
    np.random.seed(123)
    x_np = np.random.randn(batch, seq_len, num_head, dim_head).astype(np.float32)

    # Ours (needle backend)
    ours = apply_rotary_pos_emb(ndl.Tensor(x_np, device=device), base=base).numpy()

    # Hugging Face LLaMA rotary
    x_t = torch.tensor(x_np)
    # Build a minimal LlamaConfig for broad transformers compatibility.
    from transformers.models.llama.configuration_llama import LlamaConfig

    config_kwargs = dict(
        hidden_size=dim_head * num_head,
        num_attention_heads=num_head,
        max_position_embeddings=seq_len,
        rope_theta=base,
        num_hidden_layers=1,
        intermediate_size=4,
    )
    try:
        config = LlamaConfig(**config_kwargs)
    except TypeError:
        config_kwargs.pop("rope_theta", None)
        config = LlamaConfig(**config_kwargs)

    rope = LlamaRotaryEmbedding(config)

    # Ensure inv_freq matches desired base for all versions.
    rope.inv_freq = torch.pow(
        torch.tensor(base, dtype=torch.float32),
        -torch.arange(0, dim_head, 2, dtype=torch.float32) / dim_head,
    )

    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).repeat(batch, 1)
    try:
        cos, sin = rope(x_t, position_ids=position_ids)
    except TypeError:
        cos, sin = rope(x_t, seq_len=seq_len)
    hf_q, _ = hf_apply_rotary_pos_emb(x_t, x_t, cos, sin, None, unsqueeze_dim=2)
    hf_out = hf_q.detach().cpu().numpy()

    np.testing.assert_allclose(ours, hf_out, rtol=1e-5, atol=1e-6)
