import os
import sys

# Use numpy backend by default for portability; disable data import to avoid backend build requirements.
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


def rotate_half_np(x: np.ndarray) -> np.ndarray:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return np.concatenate([-x2, x1], axis=-1)


def numpy_rope(x: np.ndarray, base: float = 10000.0) -> np.ndarray:
    batch, seq_len, num_head, dim_head = x.shape
    half_dim = dim_head // 2
    inv_freq = base ** (-np.arange(0, half_dim, dtype=np.float32) / half_dim)
    freqs = np.outer(np.arange(seq_len, dtype=np.float32), inv_freq)
    cos = np.cos(freqs)
    sin = np.sin(freqs)
    cos = np.concatenate([cos, cos], axis=-1)[None, :, None, :]
    sin = np.concatenate([sin, sin], axis=-1)[None, :, None, :]

    return x * cos + rotate_half_np(x) * sin


@pytest.mark.parametrize("device", _DEVICES, ids=_DEVICE_IDS)
def test_apply_rotary_matches_numpy(device):
    base = 10000.0
    x_np = np.array(
        [
            [
                [[1.0, 2.0, 3.0, 4.0]],
                [[-1.0, -2.0, -3.0, -4.0]],
            ]
        ],
        dtype=np.float32,
    )  # shape (1, 2, 1, 4)

    out = apply_rotary_pos_emb(ndl.Tensor(x_np, device=device), base=base).numpy()
    expected = numpy_rope(x_np, base=base)
    np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("device", _DEVICES, ids=_DEVICE_IDS)
def test_attention_layer_uses_rope_with_identity_weights(device):
    base = 10000.0
    batch, seq_len, dim = 1, 3, 2
    num_head, dim_head = 1, 2
    x_np = np.array(
        [[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]],
        dtype=np.float32,
    )

    layer = nn.AttentionLayer(
        q_features=dim,
        num_head=num_head,
        dim_head=dim_head,
        dropout=0.0,
        causal=False,
        device=device,
        dtype="float32",
        rope_base=base,
    )

    # Remove normalization effects and set projections to identity.
    layer.prenorm_q = nn.Identity()
    layer.prenorm_k = nn.Identity()
    layer.prenorm_v = nn.Identity()

    eye = np.eye(dim, dtype=np.float32)
    layer.q_projection.weight = nn.Parameter(ndl.Tensor(eye, device=device))
    layer.k_projection.weight = nn.Parameter(ndl.Tensor(eye, device=device))
    layer.v_projection.weight = nn.Parameter(ndl.Tensor(eye, device=device))
    layer.out_projection.weight = nn.Parameter(
        ndl.Tensor(np.eye(dim_head, dtype=np.float32), device=device)
    )

    out = layer(ndl.Tensor(x_np, device=device)).numpy()

    # Manual attention with RoPE applied to q/k.
    qkv = x_np.reshape(batch, seq_len, num_head, dim_head)
    q_rot = numpy_rope(qkv, base=base)
    k_rot = numpy_rope(qkv, base=base)
    v = qkv

    q = np.transpose(q_rot, (0, 2, 1, 3))
    k = np.transpose(k_rot, (0, 2, 1, 3))
    v_t = np.transpose(v, (0, 2, 1, 3))

    logits = np.matmul(q, np.transpose(k, (0, 1, 3, 2))) / np.sqrt(dim_head)
    logits = logits - logits.max(axis=-1, keepdims=True)
    probs = np.exp(logits)
    probs = probs / probs.sum(axis=-1, keepdims=True)

    attn_out = np.matmul(probs, v_t)
    expected = np.transpose(attn_out, (0, 2, 1, 3)).reshape(batch, seq_len, dim_head)

    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-6)
