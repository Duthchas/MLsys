from typing import Optional, Any, Union
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        axis = Z.ndim - 1
        m = array_api.max(Z, axis=axis, keepdims=True)
        Z_shift = Z - m
        lse = array_api.log(array_api.sum(array_api.exp(Z_shift), axis=axis, keepdims=True)) + m
        return Z - lse
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        x = node.inputs[0]
        axis = len(x.shape) - 1
        lse = logsumexp(x, axes=(axis,))
        target_shape = x.shape[:-1] + (1,)
        lse_b = broadcast_to(reshape(lse, target_shape), x.shape)
        soft = exp(x - lse_b)
        sum_g = summation(out_grad, axes=(axis,))
        sum_g_b = broadcast_to(reshape(sum_g, target_shape), x.shape)
        return out_grad - soft * sum_g_b
        ### END YOUR SOLUTION


def logsoftmax(a: Tensor) -> Tensor:
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None) -> None:
        self.axes = axes

    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        if self.axes is None:
            axes_tuple = tuple(range(Z.ndim))
        else:
            axes_tuple = self.axes if isinstance(self.axes, tuple) else (self.axes,)

        m_keep = Z
        for ax in axes_tuple:
            m_keep = m_keep.max(axis=ax, keepdims=True)

        Z_shift = Z - m_keep.broadcast_to(Z.shape)

        S_keep = Z_shift.exp()
        for ax in axes_tuple:
            S_keep = S_keep.sum(axis=ax, keepdims=True)

        out_keep = S_keep.log() + m_keep

        if self.axes is None:
            return out_keep
        out_shape = tuple(s for i, s in enumerate(Z.shape) if i not in axes_tuple)
        return out_keep.reshape(out_shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        (Z,) = node.inputs
        if self.axes is None:
            axes_tuple = tuple(range(len(Z.shape)))
        else:
            axes_tuple = self.axes if isinstance(self.axes, tuple) else (self.axes,)
    
        e = exp(Z)
    
        S = e
        for ax in axes_tuple:
            S = summation(S, axes=ax)
    
        keep_shape = tuple(1 if i in axes_tuple else Z.shape[i] for i in range(len(Z.shape)))
    
        S_keep = reshape(S, keep_shape)
        S_b = broadcast_to(S_keep, Z.shape)
    
        if self.axes is None:
            og_keep = reshape(out_grad, (1,) * len(Z.shape))
        else:
            og_keep = reshape(out_grad, keep_shape)
        og_b = broadcast_to(og_keep, Z.shape)
    
        return og_b * e / S_b
        ### END YOUR SOLUTION


def logsumexp(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    return LogSumExp(axes=axes)(a)