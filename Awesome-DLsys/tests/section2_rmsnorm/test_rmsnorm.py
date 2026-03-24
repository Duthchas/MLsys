import os
import sys

# Prefer portable numpy backend for local testing unless explicitly overridden.
os.environ.setdefault("NEEDLE_BACKEND", "np")
# Disable optional data imports that rely on compiled backends.
os.environ.setdefault("NEEDLE_DISABLE_DATA", "1")

sys.path.append("./python")

import numpy as np
import pytest
import needle as ndl
import needle.nn as nn


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


def numpy_rmsnorm(x: np.ndarray, eps: float) -> np.ndarray:
    rms = np.sqrt(np.mean(np.square(x), axis=1, keepdims=True) + eps)
    return x / rms


@pytest.mark.parametrize("device", _DEVICES, ids=_DEVICE_IDS)
def test_rmsnorm_matches_numpy(device):
    eps = 1e-5
    x = np.array(
        [
            [1.0, -2.0, 3.0, -4.0],
            [0.25, -0.5, 0.5, -0.25],
        ],
        dtype=np.float32,
    )

    norm = nn.RMSNorm1d(4, eps=eps, device=device)
    out = norm(ndl.Tensor(x, device=device)).numpy()

    expected = numpy_rmsnorm(x, eps)
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("device", _DEVICES, ids=_DEVICE_IDS)
def test_rmsnorm_scales_with_weight(device):
    eps = 1e-6
    norm = nn.RMSNorm1d(3, eps=eps, device=device)

    scale = 2.5
    norm.weight = nn.Parameter(
        ndl.init.ones(3, device=device, dtype="float32") * scale
    )

    x = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    out = norm(ndl.Tensor(x, device=device)).numpy()

    expected = numpy_rmsnorm(x, eps) * scale
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("device", _DEVICES, ids=_DEVICE_IDS)
def test_attention_and_transformer_use_rmsnorm(device):
    batch, seq_len, dim = 2, 4, 6
    np.random.seed(42)

    attention = nn.AttentionLayer(
        dim,
        num_head=2,
        dim_head=3,
        dropout=0.0,
        causal=False,
        device=device,
    )

    assert isinstance(attention.prenorm_q, nn.RMSNorm1d)
    assert isinstance(attention.prenorm_k, nn.RMSNorm1d)
    assert isinstance(attention.prenorm_v, nn.RMSNorm1d)

    sample = np.random.randn(batch, seq_len, dim).astype(np.float32)
    attn_out = attention(
        ndl.Tensor(sample, device=device),
        ndl.Tensor(sample, device=device),
        ndl.Tensor(sample, device=device),
    ).numpy()
    assert attn_out.shape == (batch, seq_len, dim)
    assert np.isfinite(attn_out).all()

    transformer = nn.TransformerLayer(
        dim,
        num_head=2,
        dim_head=3,
        hidden_size=8,
        dropout=0.0,
        causal=False,
        device=device,
    )

    assert isinstance(transformer.ff_norm, nn.RMSNorm1d)

    transformer_out = transformer(ndl.Tensor(sample, device=device)).numpy()
    assert transformer_out.shape == (batch, seq_len, dim)
    assert np.isfinite(transformer_out).all()
