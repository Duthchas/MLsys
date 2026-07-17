from __future__ import annotations

from collections.abc import Iterable
from math import cos, pi

import torch


def clip_gradients(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Clip the global L2 norm of parameter gradients in place."""
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not gradients:
        return

    total_norm = torch.sqrt(sum(torch.sum(gradient * gradient) for gradient in gradients))
    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + 1e-6)
        with torch.no_grad():
            for gradient in gradients:
                gradient.mul_(scale)


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """Return the learning rate for a linear-warmup cosine-decay schedule."""
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters

    if it <= cosine_cycle_iters:
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        cosine_decay = 0.5 * (1 + cos(pi * progress))
        return min_learning_rate + cosine_decay * (max_learning_rate - min_learning_rate)

    return min_learning_rate


class AdamW(torch.optim.Optimizer):
    """Adam optimizer with decoupled weight decay."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")

                state = self.state[parameter]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)

                state["step"] += 1
                step = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                gradient = parameter.grad

                # AdamW decouples weight decay from the gradient/moment update.
                parameter.mul_(1 - lr * weight_decay)

                exp_avg.mul_(beta1).add_(gradient, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)

                step_size = lr * ((1 - beta2**step) ** 0.5) / (1 - beta1**step)
                parameter.addcdiv_(exp_avg, exp_avg_sq.sqrt().add_(eps), value=-step_size)

        return loss
