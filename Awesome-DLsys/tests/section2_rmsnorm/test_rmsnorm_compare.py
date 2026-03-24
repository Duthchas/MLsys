"""
Comparison/benchmark tests for Section 2 RMSNorm against Hugging Face LLaMA RMSNorm.
"""
import os
import sys
import time

import numpy as np
import pytest

# Prefer portable numpy backend for local testing unless explicitly overridden.
os.environ.setdefault("NEEDLE_BACKEND", "np")
# Disable optional data imports that rely on compiled backends.
os.environ.setdefault("NEEDLE_DISABLE_DATA", "1")

sys.path.append("./python")

import needle as ndl  # noqa: E402
import needle.nn as nn  # noqa: E402

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
    from transformers.models.llama.modeling_llama import LlamaRMSNorm

    _HAS_TRANSFORMERS = True
except Exception:
    _HAS_TRANSFORMERS = False


@pytest.mark.skipif(not _HAS_TRANSFORMERS, reason="transformers not installed")
@pytest.mark.parametrize("device", _DEVICES, ids=_DEVICE_IDS)
def test_rmsnorm_matches_llama_impl(device):
    np.random.seed(123)
    dim = 16
    batch = 3
    eps = 1e-6  # match HF default

    x_np = np.random.randn(batch, dim).astype(np.float32)

    ours = nn.RMSNorm1d(dim, eps=eps, device=device)
    hf = LlamaRMSNorm(dim, eps=eps)

    # Align weights so outputs are comparable.
    hf.weight = torch.nn.Parameter(torch.tensor(ours.weight.numpy()))

    out_ours = ours(ndl.Tensor(x_np, device=device)).numpy()
    out_hf = hf(torch.tensor(x_np)).detach().cpu().numpy()

    np.testing.assert_allclose(out_ours, out_hf, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not _HAS_TRANSFORMERS, reason="transformers not installed")
@pytest.mark.parametrize("device", _DEVICES, ids=_DEVICE_IDS)
def test_rmsnorm_forward_latency_small(device):
    np.random.seed(7)
    dim = 64
    batch = 32
    eps = 1e-6

    x_np = np.random.randn(batch, dim).astype(np.float32)

    ours = nn.RMSNorm1d(dim, eps=eps, device=device)
    hf = LlamaRMSNorm(dim, eps=eps)
    hf.weight = torch.nn.Parameter(torch.tensor(ours.weight.numpy()))

    # Warmup
    _ = ours(ndl.Tensor(x_np, device=device))
    _ = hf(torch.tensor(x_np))

    def time_fn(fn):
        start = time.perf_counter()
        _ = fn()
        return time.perf_counter() - start

    ours_time = time_fn(lambda: ours(ndl.Tensor(x_np, device=device)))
    hf_time = time_fn(lambda: hf(torch.tensor(x_np)))

    # Assert both run and outputs stay close; timing is informational.
    out_ours = ours(ndl.Tensor(x_np, device=device)).numpy()
    out_hf = hf(torch.tensor(x_np)).detach().cpu().numpy()
    np.testing.assert_allclose(out_ours, out_hf, rtol=1e-5, atol=1e-6)

    # Allow wide margin; this is a smoke benchmark ensuring execution succeeds.
    assert ours_time > 0 and hf_time > 0
