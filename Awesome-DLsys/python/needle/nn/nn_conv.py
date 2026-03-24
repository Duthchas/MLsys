"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import math
from .nn_basic import Parameter, Module


class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        ### BEGIN YOUR SOLUTION
        self.device = device
        self.dtype = dtype
        self.padding = kernel_size // 2

        weight_shape = (kernel_size, kernel_size, in_channels, out_channels)
        weight = init.kaiming_uniform(
            in_channels,
            out_channels,
            shape=weight_shape,
            nonlinearity="relu",
            device=device,
            dtype=dtype,
        )
        self.weight = Parameter(weight)

        if bias:
            bound = 1.0 / math.sqrt(in_channels * kernel_size * kernel_size)
            bias_tensor = init.rand(
                out_channels,
                low=-bound,
                high=bound,
                device=device,
                dtype=dtype,
            )
            self.bias = Parameter(bias_tensor)
        else:
            self.bias = None
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # Convert NCHW -> NHWC
        y = ops.transpose(x, axes=(1, 2))
        y = ops.transpose(y, axes=(2, 3))

        y = ops.conv(y, self.weight, stride=self.stride, padding=self.padding)
        if self.bias is not None:
            bias = ops.reshape(self.bias, (1, 1, 1, self.out_channels))
            y = y + ops.broadcast_to(bias, y.shape)

        # Convert back to NCHW
        y = ops.transpose(y, axes=(2, 3))
        y = ops.transpose(y, axes=(1, 2))
        return y
        ### END YOUR SOLUTION
