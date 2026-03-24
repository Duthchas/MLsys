"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from math import prod

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND
from .ops_tuple import *


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return array_api.power(a, b)
        ### END YOUR SOLUTION
        
    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        # d/da: b * a^(b-1)
        b_minus_1 = add_scalar(b, -1)
        da = out_grad * b * power(a, b_minus_1)
        # d/db: a^b * log(a)
        db = out_grad * power(a, b) * log(a)
        return da, db
        ### END YOUR SOLUTION


def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return array_api.power(a, self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, = node.inputs
        c = self.scalar
        # c * a^(c-1)
        base = power_scalar(a, c - 1)
        coeff = mul_scalar(base, c)
        return out_grad * coeff
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.divide(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        da = out_grad / b
        db = negate(out_grad * a / power_scalar(b, 2))
        return da, db
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.divide(a, self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.axes is None:
            if a.ndim < 2:
                return a
            ax1, ax2 = -1, -2
        else:
            ax1, ax2 = self.axes
        return array_api.swapaxes(a, ax1, ax2)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        axes = self.axes
        return transpose(out_grad, axes=axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        shape = list(self.shape)
        neg_idx = None
        unknown = 1
        for i, dim in enumerate(shape):
            if dim == -1:
                assert neg_idx is None, "Only one dimension can be inferred"
                neg_idx = i
            else:
                unknown *= dim
        total = a.size
        if neg_idx is not None:
            assert total % unknown == 0, "Inferred dimension mismatch"
            shape[neg_idx] = total // unknown
        assert prod(a.shape) == prod(shape), "Product of dimensions must remain the same."
        if hasattr(a, "is_compact") and not a.is_compact():
            a = a.compact()
        return array_api.reshape(a, tuple(shape))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, = node.inputs
        return reshape(out_grad, a.shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.broadcast_to(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, = node.inputs
        in_shape = a.shape
        out_shape = self.shape

        axes = []
        lead = len(out_shape) - len(in_shape)
        if lead > 0:
            axes.extend(range(0, lead))

        for i, (s_in, s_out) in enumerate(zip(in_shape, out_shape[lead:])):
            if s_in == 1 and s_out != 1:
                axes.append(lead + i)
        if axes:
            out_grad = summation(out_grad, axes=tuple(axes))

        if out_grad.shape != in_shape:
            out_grad = reshape(out_grad, in_shape)
        return out_grad
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.sum(a, axis=self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, = node.inputs
        in_shape = a.shape
        if self.axes is None:
            resh = reshape(out_grad, (1,) * len(in_shape))
            return broadcast_to(resh, in_shape)
        axes = self.axes if isinstance(self.axes, tuple) else (self.axes,)
        tmp_shape = list(in_shape)
        for ax in axes:
            tmp_shape[ax] = 1
        resh = reshape(out_grad, tuple(tmp_shape))
        return broadcast_to(resh, in_shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.matmul(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        A, B = node.inputs
        # raw grads
        dA = matmul(out_grad, transpose(B, axes=(-1, -2)))
        dB = matmul(transpose(A, axes=(-1, -2)), out_grad)

        def reduce_to_shape(g, shape):
            while len(g.shape) > len(shape):
                g = summation(g, axes=(0,))
            axes = tuple(i for i, (gs, s) in enumerate(zip(g.shape, shape)) if s == 1 and gs != 1)
            if axes:
                g = summation(g, axes=axes)
            if g.shape != shape:
                g = reshape(g, shape)
            return g

        dA = reduce_to_shape(dA, A.shape)
        dB = reduce_to_shape(dB, B.shape)
        return dA, dB
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.negative(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return negate(out_grad)
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, = node.inputs
        return out_grad / a
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, = node.inputs
        return out_grad * exp(a)
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        y = node.realize_cached_data()
        if array_api is numpy:
            mask_array = (y > 0).astype(y.dtype)
        else:
            mask_array = array_api.array(y > 0, device=y.device, dtype=y.dtype)
        mask = Tensor.make_const(mask_array)
        return out_grad * mask
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.tanh(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        y = node.realize_cached_data()
        one = array_api.full(y.shape, 1.0, device=y.device)
        one_minus_y2 = one - y * y
        return out_grad * Tensor.make_const(one_minus_y2)
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        ### BEGIN YOUR SOLUTION
        xs = list(args)
        base_shape = xs[0].shape
        for x in xs:
            assert x.shape == base_shape
        rank = len(base_shape)
        axis = self.axis if self.axis >= 0 else self.axis + rank + 1
        out_shape = list(base_shape)
        out_shape.insert(axis, len(xs))
        out = array_api.full(tuple(out_shape), 0.0, device=xs[0].device)
        resh_shape = list(base_shape)
        resh_shape.insert(axis, 1)
        resh_shape = tuple(resh_shape)
        for i, x in enumerate(xs):
            x_view = array_api.reshape(x, resh_shape)
            sl = [slice(None)] * len(out_shape)
            sl[axis] = slice(i, i + 1)
            out[tuple(sl)] = x_view
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return split(out_grad, self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        axis = self.axis if self.axis >= 0 else self.axis + A.ndim
        k = A.shape[axis]
        base_shape = list(A.shape)
        del base_shape[axis]
        base_shape = tuple(base_shape)
        outs = []
        for i in range(k):
            sl = [slice(None)] * A.ndim
            sl[axis] = slice(i, i + 1)
            view = A[tuple(sl)]
            view = view.compact()
            outs.append(array_api.reshape(view, base_shape))
        return tuple(outs)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return stack(list(out_grad), self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        axes = self.axes
        if axes is None:
            axes_tuple = tuple(range(a.ndim))
        elif isinstance(axes, (tuple, list)):
            axes_tuple = tuple(axes)
        else:
            axes_tuple = (axes,)
        return a.flip(axes_tuple)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (flip(out_grad, self.axes),)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        dilation = self.dilation
        if dilation < 0:
            raise ValueError("Dilation must be non-negative")

        if self.axes is None:
            axes = tuple(range(a.ndim))
        elif isinstance(self.axes, (tuple, list)):
            axes = tuple(self.axes)
        else:
            axes = (self.axes,)

        ndim = a.ndim
        axes_norm: list[int] = []
        for ax in axes:
            ax = ax + ndim if ax < 0 else ax
            if ax < 0 or ax >= ndim:
                continue
            if ax not in axes_norm:
                axes_norm.append(ax)

        if dilation == 0 or len(axes_norm) == 0:
            return a

        step = dilation + 1
        orig_shape = a.shape
        new_shape = list(orig_shape)
        for ax in axes_norm:
            new_shape[ax] = orig_shape[ax] * (dilation + 1)

        out = array_api.full(tuple(new_shape), 0.0, device=a.device)
        idx = []
        axes_set = set(axes_norm)
        for dim in range(ndim):
            if dim in axes_set:
                idx.append(slice(0, None, step))
            else:
                idx.append(slice(None))
        out[tuple(idx)] = a
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (undilate(out_grad, self.axes, self.dilation),)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        dilation = self.dilation
        if dilation < 0:
            raise ValueError("Dilation must be non-negative")

        if self.axes is None:
            axes = tuple(range(a.ndim))
        elif isinstance(self.axes, (tuple, list)):
            axes = tuple(self.axes)
        else:
            axes = (self.axes,)

        ndim = a.ndim
        axes_norm: list[int] = []
        for ax in axes:
            ax = ax + ndim if ax < 0 else ax
            if ax < 0 or ax >= ndim:
                continue
            if ax not in axes_norm:
                axes_norm.append(ax)

        if dilation == 0 or len(axes_norm) == 0:
            return a

        step = dilation + 1
        axes_set = set(axes_norm)
        idx = []
        for dim in range(ndim):
            if dim in axes_set:
                idx.append(slice(0, None, step))
            else:
                idx.append(slice(None))
        return a[tuple(idx)].compact()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (dilate(out_grad, self.axes, self.dilation),)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        stride = self.stride
        padding = self.padding

        assert isinstance(stride, int)
        assert isinstance(padding, int)

        if hasattr(A, "is_compact") and not A.is_compact():
            A = A.compact()
        if hasattr(B, "is_compact") and not B.is_compact():
            B = B.compact()

        if padding > 0:
            A = A.pad(((0, 0), (padding, padding), (padding, padding), (0, 0)))

        N, H, W, C_in = A.shape
        k1, k2, C_in_w, C_out = B.shape

        assert k1 == k2, "Kernel must be square"
        assert C_in == C_in_w, "Input and weight channel mismatch"

        out_h = (H - k1) // stride + 1
        out_w = (W - k1) // stride + 1

        sN, sH, sW, sC = A.strides
        window_shape = (N, out_h, out_w, k1, k1, C_in)
        window_strides = (sN, sH * stride, sW * stride, sH, sW, sC)

        patches = A.as_strided(window_shape, window_strides).compact()
        patches2d = patches.reshape((N * out_h * out_w, k1 * k1 * C_in))

        weight_mat = B.reshape((k1 * k1 * C_in, C_out))
        out = array_api.matmul(patches2d, weight_mat)
        return out.reshape((N, out_h, out_w, C_out))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        X, W = node.inputs
        stride = self.stride
        padding = self.padding

        grad_out = out_grad
        grad_input_tensor = grad_out
        if stride > 1:
            grad_input_tensor = dilate(grad_out, axes=(1, 2), dilation=stride - 1)

        # Gradient w.r.t input X
        W_flip = flip(W, axes=(0, 1))
        W_flip = transpose(W_flip, axes=(2, 3))
        pad_input = W.shape[0] - 1 - padding
        assert pad_input >= 0
        grad_input = conv(grad_input_tensor, W_flip, stride=1, padding=pad_input)

        # Gradient w.r.t weights W via im2col-style accumulation
        X_arr = X.realize_cached_data()
        if padding > 0:
            X_arr = X_arr.pad(((0, 0), (padding, padding), (padding, padding), (0, 0)))
        X_arr = X_arr.compact()
        stride_val = self.stride
        N, H_pad, W_pad, C_in = X_arr.shape
        k = W.shape[0]
        C_out = W.shape[-1]

        grad_arr = grad_out.realize_cached_data().compact()
        out_h = grad_arr.shape[1]
        out_w = grad_arr.shape[2]

        sN, sH, sW, sC = X_arr.strides
        patches = X_arr.as_strided(
            (N, out_h, out_w, k, k, C_in),
            (sN, sH * stride_val, sW * stride_val, sH, sW, sC),
        ).compact()
        patches2d = patches.reshape((N * out_h * out_w, k * k * C_in)).compact()
        grad2d = grad_arr.reshape((N * out_h * out_w, C_out)).compact()
        patches2d_T = patches2d.permute((1, 0)).compact()
        grad_weight_mat = array_api.matmul(patches2d_T, grad2d)
        grad_weight_arr = grad_weight_mat.reshape((k, k, C_in, C_out)).compact()
        grad_weight = Tensor.make_const(grad_weight_arr, requires_grad=False)

        return grad_input, grad_weight
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)
