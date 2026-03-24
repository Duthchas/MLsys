from typing import Optional, Any, Union
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        log_sum_exp = LogSumExp(axes=(-1,)).compute(Z)
        
        shape = list(Z.shape)
        shape[-1] = 1

        log_sum_exp_reshaped = log_sum_exp.reshape(tuple(shape))
        return Z - log_sum_exp_reshaped
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]
        softmax_Z = exp(node)

        sum_out_grad = out_grad.sum(axes=(-1,))
        shape = list(Z.shape)
        shape[-1] = 1
        sum_out_grad_reshaped = sum_out_grad.reshape(tuple(shape))
        return out_grad - softmax_Z * sum_out_grad_reshaped
        ### END YOUR SOLUTION


def logsoftmax(a: Tensor) -> Tensor:
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None) -> None:
        self.axes = axes

    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        max_Z = Z.max(axis=self.axes, keepdims=True)
        max_Z_reduce = Z.max(axis=self.axes, keepdims=False)
        sum_ = array_api.sum(array_api.exp(Z - max_Z), self.axes, keepdims=False)
        
        return array_api.log(sum_) + max_Z_reduce
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]
        log_sum_exp_Z = node

        new_shape = list(Z.shape)
        if self.axes is not None:
          for axis in self.axes:
            new_shape[axis] = 1
        else:
          new_shape = [1] * len(Z.shape)

        log_sum_exp_Z_reshaped = log_sum_exp_Z.reshape(tuple(new_shape))
        
        softmax_val = exp(Z - log_sum_exp_Z_reshaped)

        out_grad_reshaped = out_grad.reshape(tuple(new_shape))

        return softmax_val * out_grad_reshaped
        ### END YOUR SOLUTION


def logsumexp(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    return LogSumExp(axes=axes)(a)