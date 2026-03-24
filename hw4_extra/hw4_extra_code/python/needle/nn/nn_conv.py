"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
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
        fan_in = in_channels * kernel_size**2
        weight_shape = (kernel_size, kernel_size, in_channels, out_channels)
        weight_tensor = init.kaiming_uniform(fan_in, out_channels, shape = weight_shape)
        self.weight = Parameter(weight_tensor, device=device)

        self.padding = (kernel_size - 1) // 2

        if bias:
          bound = 1 / np.sqrt(fan_in)
          bias_tensor = init.rand(out_channels, low=-bound, high=bound)
          self.bias = Parameter(bias_tensor, device=device)
        else:
          self.bias = None
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        x_nhwc = x.transpose((1,2)).transpose((2,3))

        conv_out = ops.conv(x_nhwc, self.weight, stride=self.stride, padding=self.padding)

        if self.bias:
          bias_reshaped = self.bias.reshape((1, 1, 1, self.out_channels))
          conv_out = conv_out + bias_reshaped.broadcast_to(conv_out.shape)
        
        return conv_out.transpose((2,3)).transpose((1,2))
        ### END YOUR SOLUTION