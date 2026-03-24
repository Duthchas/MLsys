"""The module.
"""
from typing import Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> list[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> list["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self) -> None:
        self.training = True

    def parameters(self) -> list[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> list["Module"]:
        return _child_modules(self.__dict__)

    def eval(self) -> None:
        self.training = False
        for m in self._children():
            m.training = False

    def train(self) -> None:
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.device = device
        self.dtype = dtype

        W = init.kaiming_uniform(in_features, out_features, nonlinearity="relu", device=device, dtype=dtype)
        self.weight = Parameter(W)

        if bias:
            b2d = init.kaiming_uniform(out_features, 1, nonlinearity="relu", device=device, dtype=dtype)
            b = ops.reshape(b2d, (1, out_features))
            self.bias = Parameter(b)
        else:
            self.bias = None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        Y = ops.matmul(X, self.weight)
        if self.bias is not None:
            Y = Y + ops.broadcast_to(self.bias, Y.shape)
        return Y
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # (N, d1, d2, ..., dk) -> (N, d1*d2*...*dk)
        N = X.shape[0]
        feat = 1
        for s in X.shape[1:]:
            feat *= s
        return ops.reshape(X, (N, feat))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION

class Sequential(Module):
    def __init__(self, *modules: Module) -> None:
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for m in self.modules:
            x = m(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        N, C = logits.shape[0], logits.shape[1]
        lse = ops.logsumexp(logits, axes=(1,))
        y_one_hot = init.one_hot(C, y, device=getattr(logits, "device", None), dtype=getattr(logits, "dtype", "float32"))
        zy = ops.summation(logits * y_one_hot, axes=(1,))
        loss_vec = lse - zy
        return ops.summation(loss_vec) / N 
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.device = device
        self.dtype = dtype
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))  # gamma
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype)) # beta

        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        N = x.shape[0]
        if self.training:
            mean = ops.summation(x, axes=(0,)) / N
            mean_b = ops.broadcast_to(ops.reshape(mean, (1, self.dim)), x.shape)
            x_centered = x - mean_b
            var = ops.summation(x_centered * x_centered, axes=(0,)) / N

            inv_std = (var + self.eps) ** (-0.5)
            inv_std_b = ops.broadcast_to(ops.reshape(inv_std, (1, self.dim)), x.shape)
            x_hat = x_centered * inv_std_b

            m = self.momentum
            self.running_mean = (1 - m) * self.running_mean + m * mean
            self.running_var = (1 - m) * self.running_var  + m * var
        else:
            mean = self.running_mean
            var  = self.running_var
            mean_b = ops.broadcast_to(ops.reshape(mean, (1, self.dim)), x.shape)
            inv_std = (var + self.eps) ** (-0.5)
            inv_std_b = ops.broadcast_to(ops.reshape(inv_std, (1, self.dim)), x.shape)
            x_hat = (x - mean_b) * inv_std_b

        gamma_b = ops.broadcast_to(ops.reshape(self.weight, (1, self.dim)), x.shape)
        beta_b  = ops.broadcast_to(ops.reshape(self.bias, (1, self.dim)), x.shape)
        return gamma_b * x_hat + beta_b
        ### END YOUR SOLUTION


class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))



class LayerNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.device = device
        self.dtype = dtype
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias   = Parameter(init.zeros(dim, device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        N = x.shape[0]
        mean = ops.summation(x, axes=(1,)) / self.dim
        mean = ops.reshape(mean, (N, 1))
        x_centered = x - ops.broadcast_to(mean, x.shape)

        var = ops.summation(x_centered * x_centered, axes=(1,)) / self.dim
        var = ops.reshape(var, (N, 1))
        inv_std = (var + self.eps) ** (-0.5)

        x_hat = x_centered * ops.broadcast_to(inv_std, x.shape)

        gamma_b = ops.broadcast_to(ops.reshape(self.weight, (1, self.dim)), x.shape)
        beta_b = ops.broadcast_to(ops.reshape(self.bias, (1, self.dim)), x.shape)
        return gamma_b * x_hat + beta_b        
        ### END YOUR SOLUTION


class RMSNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        # Root Mean Square Normalization: normalize by vector rms without centering.
        batch = x.shape[0]
        rms = ops.summation(x * x, axes=(1,)) / self.dim
        rms = ops.reshape(rms, (batch, 1))
        inv_rms = ops.power_scalar(ops.add_scalar(rms, self.eps), -0.5)

        scale = ops.broadcast_to(ops.reshape(self.weight, (1, self.dim)), x.shape)
        norm = ops.broadcast_to(inv_rms, x.shape)
        return x * norm * scale


class Dropout(Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if not self.training or self.p == 0.0:
            return x
        keep_prob = 1.0 - self.p
        mask = init.randb(*x.shape, p=keep_prob, device=getattr(x, "device", None), dtype=getattr(x, "dtype", "float32"))
        return x * mask / keep_prob
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return x + self.fn(x)
        ### END YOUR SOLUTION
