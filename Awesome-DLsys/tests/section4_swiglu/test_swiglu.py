import os
import sys

# Use numpy backend for determinism; skip data imports.
os.environ.setdefault("NEEDLE_BACKEND", "np")
os.environ.setdefault("NEEDLE_DISABLE_DATA", "1")

sys.path.append("./python")

import numpy as np
import torch
import pytest

import needle as ndl
from needle import ops
import needle.nn as nn
from needle.nn.nn_basic import Parameter

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


def _run_ff_only(layer: nn.TransformerLayer, x_np: np.ndarray, device) -> np.ndarray:
    """Run only the feed-forward path (no residual) using layer modules."""
    b, s, d = x_np.shape
    x = ndl.Tensor(x_np, device=device)
    norm_in = ops.reshape(x, (b * s, d))
    normed = layer.ff_norm(norm_in)

    ff1 = layer.ff_linear1(normed)
    ff1 = ops.reshape(ff1, (b, s, 2, layer.hidden_size))
    u, v = ops.split(ff1, axis=2)
    u = ops.reshape(u, (b, s, layer.hidden_size))
    v = ops.reshape(v, (b, s, layer.hidden_size))

    exp_neg_v = ops.exp(ops.negate(v))
    denom = ops.add_scalar(exp_neg_v, 1.0)
    sigmoid_v = ops.power_scalar(denom, -1)
    swish_v = v * sigmoid_v

    hidden = u * swish_v
    hidden = ops.reshape(hidden, (b * s, layer.hidden_size))
    hidden = layer.ff_dropout(hidden)
    hidden = layer.ff_linear2(hidden)
    hidden = layer.ff_output_dropout(hidden)

    return ops.reshape(hidden, (b, s, d)).numpy()


@pytest.mark.parametrize("device", _DEVICES, ids=_DEVICE_IDS)
def test_swiglu_matches_numpy(device):
    np.random.seed(0)
    batch, seq, d_model, hidden = 1, 2, 4, 3
    x_np = np.random.randn(batch, seq, d_model).astype(np.float32)

    layer = nn.TransformerLayer(
        d_model,
        num_head=1,
        dim_head=2,
        hidden_size=hidden,
        dropout=0.0,
        causal=False,
        device=device,
    )

    w1 = np.random.randn(d_model, 2 * hidden).astype(np.float32)
    w2 = np.random.randn(hidden, d_model).astype(np.float32)
    layer.ff_linear1.weight = Parameter(ndl.Tensor(w1, device=device))
    layer.ff_linear1.bias = Parameter(
        ndl.Tensor(np.zeros((1, 2 * hidden), dtype=np.float32), device=device)
    )
    layer.ff_linear2.weight = Parameter(ndl.Tensor(w2, device=device))
    layer.ff_linear2.bias = Parameter(
        ndl.Tensor(np.zeros((1, d_model), dtype=np.float32), device=device)
    )

    # Manual numpy computation of SwiGLU FFN (without residual).
    eps = 1e-5
    x_flat = x_np.reshape(batch * seq, d_model)
    rms = np.sqrt(np.sum(x_flat * x_flat, axis=1, keepdims=True) / d_model + eps)
    normed = x_flat / rms

    ff1 = normed @ w1
    ff1 = ff1.reshape(batch, seq, 2, hidden)
    u = ff1[:, :, 0, :]
    v = ff1[:, :, 1, :]
    swish_v = v * (1.0 / (1.0 + np.exp(-v)))
    hidden_np = u * swish_v
    hidden_flat = hidden_np.reshape(batch * seq, hidden)
    out_np = hidden_flat @ w2
    out_np = out_np.reshape(batch, seq, d_model)

    out = _run_ff_only(layer, x_np, device)
    np.testing.assert_allclose(out, out_np, rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(not _HAS_TRANSFORMERS, reason="transformers not installed")
@pytest.mark.parametrize("device", _DEVICES, ids=_DEVICE_IDS)
def test_swiglu_matches_hf_llama_mlp(device):
    from transformers.models.llama.modeling_llama import LlamaConfig, LlamaMLP, LlamaRMSNorm

    np.random.seed(123)
    batch, seq, d_model, hidden = 2, 3, 6, 8
    x_np = np.random.randn(batch, seq, d_model).astype(np.float32)

    layer = nn.TransformerLayer(
        d_model,
        num_head=1,
        dim_head=2,
        hidden_size=hidden,
        dropout=0.0,
        causal=False,
        device=device,
    )

    w1 = np.random.randn(d_model, 2 * hidden).astype(np.float32)
    w2 = np.random.randn(hidden, d_model).astype(np.float32)
    layer.ff_linear1.weight = Parameter(ndl.Tensor(w1, device=device))
    layer.ff_linear1.bias = Parameter(
        ndl.Tensor(np.zeros((1, 2 * hidden), dtype=np.float32), device=device)
    )
    layer.ff_linear2.weight = Parameter(ndl.Tensor(w2, device=device))
    layer.ff_linear2.bias = Parameter(
        ndl.Tensor(np.zeros((1, d_model), dtype=np.float32), device=device)
    )
    layer.ff_norm.weight = Parameter(
        ndl.Tensor(np.ones((d_model,), dtype=np.float32), device=device)
    )

    config = LlamaConfig(
        hidden_size=d_model,
        intermediate_size=hidden,
        num_attention_heads=1,
        num_hidden_layers=1,
        rms_norm_eps=1e-5,
    )
    hf_rms = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
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
        hf_rms.weight.copy_(torch.tensor(layer.ff_norm.weight.numpy()))

    torch_x = torch.tensor(x_np)
    hf_out = hf_mlp(hf_rms(torch_x)).detach().cpu().numpy()

    ours = _run_ff_only(layer, x_np, device)
    np.testing.assert_allclose(ours, hf_out, rtol=1e-5, atol=1e-6)
