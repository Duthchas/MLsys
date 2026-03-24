import math
from .init_basic import *
from typing import Any, Iterable, Tuple


def _calculate_fans(shape: Iterable[int]) -> Tuple[int, int]:
    dims = tuple(int(d) for d in shape)
    if len(dims) == 0:
        return 1, 1
    if len(dims) == 1:
        fan = dims[0]
        return fan, fan
    if len(dims) == 2:
        return dims[0], dims[1]
    receptive_field = 1
    for dim in dims[:-2]:
        receptive_field *= dim
    fan_in = dims[-2] * receptive_field
    fan_out = dims[-1] * receptive_field
    return fan_in, fan_out


def _resolve_shape(fan_in: int, fan_out: int, shape) -> Tuple[Tuple[int, ...], int, int]:
    if shape is None:
        resolved_shape = (fan_in, fan_out)
        fin, fout = fan_in, fan_out
    else:
        resolved_shape = tuple(shape)
        fin, fout = _calculate_fans(resolved_shape)
    return resolved_shape, fin, fout


def xavier_uniform(fan_in: int, fan_out: int, gain: float = 1.0, shape=None, **kwargs: Any) -> "Tensor":
    ### BEGIN YOUR SOLUTION
    resolved_shape, fin, fout = _resolve_shape(fan_in, fan_out, shape)
    bound = gain * math.sqrt(6.0 / (fin + fout))
    return rand(*resolved_shape, low=-bound, high=bound, **kwargs)
    ### END YOUR SOLUTION


def xavier_normal(fan_in: int, fan_out: int, gain: float = 1.0, shape=None, **kwargs: Any) -> "Tensor":
    ### BEGIN YOUR SOLUTION
    resolved_shape, fin, fout = _resolve_shape(fan_in, fan_out, shape)
    std = gain * math.sqrt(2.0 / (fin + fout))
    return randn(*resolved_shape, mean=0.0, std=std, **kwargs)
    ### END YOUR SOLUTION


def kaiming_uniform(fan_in: int, fan_out: int, nonlinearity: str = "relu", shape=None, **kwargs: Any) -> "Tensor":
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    resolved_shape, fin, _ = _resolve_shape(fan_in, fan_out, shape)
    bound = math.sqrt(6.0 / fin)
    return rand(*resolved_shape, low=-bound, high=bound, **kwargs)
    ### END YOUR SOLUTION


def kaiming_normal(fan_in: int, fan_out: int, nonlinearity: str = "relu", shape=None, **kwargs: Any) -> "Tensor":
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    resolved_shape, fin, _ = _resolve_shape(fan_in, fan_out, shape)
    std = math.sqrt(2.0 / fin)
    return randn(*resolved_shape, mean=0.0, std=std, **kwargs)
    ### END YOUR SOLUTION
