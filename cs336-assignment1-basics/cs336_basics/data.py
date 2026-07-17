from __future__ import annotations

import numpy as np
import torch


def get_batch(
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample input and next-token sequences from a one-dimensional token dataset."""
    start_indices = np.random.randint(
        0,
        len(dataset) - context_length,
        size=batch_size,
    )

    inputs = np.stack([dataset[start : start + context_length] for start in start_indices])
    targets = np.stack([dataset[start + 1 : start + context_length + 1] for start in start_indices])

    return (
        torch.as_tensor(inputs, dtype=torch.long, device=device),
        torch.as_tensor(targets, dtype=torch.long, device=device),
    )
