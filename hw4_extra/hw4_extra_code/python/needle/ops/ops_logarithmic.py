from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_Z = Z.max(axis=self.axes, keepdims=True).broadcast_to(Z.shape)
        max_Z_reduce = Z.max(axis=self.axes, keepdims=False)
        sum_ = array_api.sum(array_api.exp(Z - max_Z), self.axes, keepdims=False)
        
        return array_api.log(sum_) + max_Z_reduce
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]

        new_shape = list(Z.shape)
        if self.axes is not None:
          for axis in self.axes:
            new_shape[axis] = 1
        else:
          new_shape = [1] * len(Z.shape)

        node_reshaped = node.reshape(tuple(new_shape))
        
        softmax_val = exp(Z - node_reshaped.broadcast_to(Z.shape))

        out_grad_reshaped = out_grad.reshape(tuple(new_shape))

        return out_grad_reshaped.broadcast_to(Z.shape) * softmax_val
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

