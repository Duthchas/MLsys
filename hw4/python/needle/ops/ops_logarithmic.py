from typing import Optional, Any, Union
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

class LogSoftmax(TensorOp):
    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a: Tensor) -> Tensor:
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None) -> None:
        self.axes = axes

    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        max_Z = Z.max(axis=self.axes, keepdims=True).broadcast_to(Z.shape)
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
          if isinstance(self.axes, int):
            axes_to_iterate = (self.axes,) 
          else:
            axes_to_iterate = self.axes

          for axis in axes_to_iterate:
            new_shape[axis] = 1
        else:
          new_shape = [1] * len(Z.shape)

        log_sum_exp_Z_reshaped = log_sum_exp_Z.reshape(tuple(new_shape))
        
        softmax_val = exp(Z - log_sum_exp_Z_reshaped.broadcast_to(Z.shape))

        out_grad_reshaped = out_grad.reshape(tuple(new_shape))

        return out_grad_reshaped.broadcast_to(Z.shape) * softmax_val
        ### END YOUR SOLUTION


def logsumexp(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    return LogSumExp(axes=axes)(a)