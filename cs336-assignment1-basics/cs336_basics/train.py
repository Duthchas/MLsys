"""Command-line training loop for the CS336 Transformer language model.

Example:

    python -m cs336_basics.train \
        --train-data data/train.npy \
        --valid-data data/valid.npy \
        --vocab-size 10000 \
        --context-length 256 \
        --max-steps 10000 \
        --checkpoint-path checkpoints/model.pt

``.npy`` datasets are opened with ``mmap_mode="r"``. Other file extensions are
treated as raw one-dimensional arrays and opened with ``np.memmap`` using the
dtype given by ``--data-dtype``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cs336_basics.checkpoint import load_checkpoint, save_checkpoint
from cs336_basics.data import get_batch
from cs336_basics.nn import TransformerLM
from cs336_basics.optimizer import AdamW, clip_gradients, get_lr_cosine_schedule
from cs336_basics.softmax import cross_entropy


TORCH_DTYPES = {
    "float32": torch.float32,
    "float64": torch.float64,
}


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_dataset(path: str | Path, data_dtype: str) -> np.ndarray:
    """Open a token dataset without eagerly copying it into memory."""
    path = Path(path)
    if path.suffix.lower() == ".npy":
        dataset = np.load(path, mmap_mode="r")
    else:
        dataset = np.memmap(path, dtype=np.dtype(data_dtype), mode="r")

    if dataset.ndim != 1:
        raise ValueError(f"Expected a 1D token dataset, got shape {dataset.shape} for {path}")
    if len(dataset) == 0:
        raise ValueError(f"Dataset is empty: {path}")
    return dataset


def _validate_dataset(dataset: np.ndarray, name: str, vocab_size: int, context_length: int) -> None:
    if len(dataset) <= context_length:
        raise ValueError(
            f"{name} dataset has {len(dataset)} tokens, but context_length is {context_length}; "
            "at least context_length + 1 tokens are required"
        )

    minimum = int(np.min(dataset))
    maximum = int(np.max(dataset))
    if minimum < 0 or maximum >= vocab_size:
        raise ValueError(
            f"{name} dataset contains token IDs outside [0, {vocab_size}): "
            f"minimum={minimum}, maximum={maximum}"
        )


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: torch.device,
    eval_steps: int,
) -> float:
    """Estimate mean validation loss over randomly sampled validation batches."""
    model.eval()
    losses = []
    for _ in range(eval_steps):
        inputs, targets = get_batch(dataset, batch_size, context_length, device)
        logits = model(inputs)
        losses.append(cross_entropy(logits, targets))

    mean_loss = torch.stack(losses).mean().item()
    model.train()
    return mean_loss


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)

    data = parser.add_argument_group("data")
    data.add_argument("--train-data", required=True, help="Path to a .npy or raw token dataset")
    data.add_argument("--valid-data", help="Optional path to a validation .npy or raw token dataset")
    data.add_argument(
        "--data-dtype",
        default="uint16",
        help="NumPy dtype for raw datasets; ignored for .npy files (default: uint16)",
    )
    data.add_argument("--verify-data", action="store_true", help="Scan datasets and verify token IDs")

    model = parser.add_argument_group("model")
    model.add_argument("--vocab-size", type=_positive_int, required=True)
    model.add_argument("--context-length", type=_positive_int, required=True)
    model.add_argument("--d-model", type=_positive_int, default=512)
    model.add_argument("--num-layers", type=_positive_int, default=4)
    model.add_argument("--num-heads", type=_positive_int, default=16)
    model.add_argument("--d-ff", type=_positive_int, default=1344)
    model.add_argument("--rope-theta", type=float, default=10000.0)
    model.add_argument("--dtype", choices=sorted(TORCH_DTYPES), default="float32")

    optimization = parser.add_argument_group("optimization")
    optimization.add_argument("--batch-size", type=_positive_int, default=32)
    optimization.add_argument("--max-steps", type=_positive_int, default=10000)
    optimization.add_argument("--learning-rate", type=_nonnegative_float, default=3e-4)
    optimization.add_argument("--min-learning-rate", type=_nonnegative_float, default=3e-5)
    optimization.add_argument("--warmup-iters", type=int, default=100)
    optimization.add_argument("--cosine-cycle-iters", type=int)
    optimization.add_argument("--beta1", type=float, default=0.9)
    optimization.add_argument("--beta2", type=float, default=0.95)
    optimization.add_argument("--eps", type=_nonnegative_float, default=1e-8)
    optimization.add_argument("--weight-decay", type=_nonnegative_float, default=0.1)
    optimization.add_argument("--max-grad-norm", type=_nonnegative_float, default=1.0)

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--device", default=None, help=f"Training device (default: {_default_device()})")
    runtime.add_argument("--seed", type=int, default=42)
    runtime.add_argument("--log-interval", type=_positive_int, default=10)
    runtime.add_argument("--eval-interval", type=_positive_int, default=500)
    runtime.add_argument("--eval-steps", type=_positive_int, default=20)
    runtime.add_argument("--checkpoint-path", type=Path)
    runtime.add_argument("--checkpoint-interval", type=_positive_int, default=500)
    runtime.add_argument("--resume", type=Path, help="Load model and optimizer state from this checkpoint")
    runtime.add_argument("--wandb-project", help="Optional Weights & Biases project name")
    runtime.add_argument("--wandb-run-name")

    return parser


def _log(metrics: dict[str, float], step: int, wandb_run: Any | None) -> None:
    formatted = " ".join(f"{name}={value:.6f}" for name, value in metrics.items())
    print(f"step={step} {formatted}", flush=True)
    if wandb_run is not None:
        wandb_run.log(metrics, step=step)


def train(args: argparse.Namespace) -> None:
    if args.d_model % args.num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads")
    if args.warmup_iters < 0:
        raise ValueError("warmup_iters must be non-negative")
    cosine_cycle_iters = args.cosine_cycle_iters or args.max_steps
    if cosine_cycle_iters <= args.warmup_iters:
        raise ValueError("cosine_cycle_iters must be greater than warmup_iters")
    if args.min_learning_rate > args.learning_rate:
        raise ValueError("min-learning-rate must not exceed learning-rate")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device or _default_device())
    dtype = TORCH_DTYPES[args.dtype]
    train_data = _load_dataset(args.train_data, args.data_dtype)
    valid_data = _load_dataset(args.valid_data, args.data_dtype) if args.valid_data else None

    if args.verify_data:
        _validate_dataset(train_data, "training", args.vocab_size, args.context_length)
        if valid_data is not None:
            _validate_dataset(valid_data, "validation", args.vocab_size, args.context_length)
    elif len(train_data) <= args.context_length:
        raise ValueError("training dataset must contain more than context_length tokens")
    if valid_data is not None and len(valid_data) <= args.context_length:
        raise ValueError("validation dataset must contain more than context_length tokens")

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=device,
        dtype=dtype,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    start_step = 0
    if args.resume is not None:
        start_step = load_checkpoint(args.resume, model, optimizer)
        print(f"resumed checkpoint={args.resume} step={start_step}", flush=True)

    checkpoint_path = args.checkpoint_path or args.resume
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    wandb_run = None
    if args.wandb_project:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )

    running_loss = 0.0
    running_steps = 0
    try:
        for step_index in range(start_step, args.max_steps):
            step = step_index + 1
            learning_rate = get_lr_cosine_schedule(
                it=step_index,
                max_learning_rate=args.learning_rate,
                min_learning_rate=args.min_learning_rate,
                warmup_iters=args.warmup_iters,
                cosine_cycle_iters=cosine_cycle_iters,
            )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate

            inputs, targets = get_batch(train_data, args.batch_size, args.context_length, device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = cross_entropy(logits, targets)
            loss.backward()
            if args.max_grad_norm > 0:
                clip_gradients(model.parameters(), args.max_grad_norm)
            optimizer.step()

            running_loss += loss.item()
            running_steps += 1

            should_log = step % args.log_interval == 0 or step == args.max_steps
            should_eval = valid_data is not None and (
                step % args.eval_interval == 0 or step == args.max_steps
            )
            should_checkpoint = checkpoint_path is not None and (
                step % args.checkpoint_interval == 0 or step == args.max_steps
            )

            if should_log or should_eval:
                metrics = {
                    "train/loss": running_loss / running_steps,
                    "train/learning_rate": learning_rate,
                }
                if should_eval:
                    metrics["valid/loss"] = evaluate(
                        model,
                        valid_data,
                        args.batch_size,
                        args.context_length,
                        device,
                        args.eval_steps,
                    )
                _log(metrics, step, wandb_run)
                running_loss = 0.0
                running_steps = 0

            if should_checkpoint:
                save_checkpoint(model, optimizer, step, checkpoint_path)
                print(f"saved checkpoint={checkpoint_path} step={step}", flush=True)
    finally:
        if wandb_run is not None:
            wandb_run.finish()


def main() -> None:
    args = _build_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
