"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for param in self.params:
          decay = param.grad.data + self.weight_decay * param.data
          grad = self.momentum * self.u.get(param, 0) + (1 - self.momentum) * decay
          self.u[param] = ndl.Tensor(grad, dtype=param.dtype)
          param.data = param.data - self.lr * self.u[param].data
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for param in self.params:
          decay = param.grad.data + self.weight_decay * param.data

          m_prev = self.m.get(param, 0)
          v_prev = self.v.get(param, 0)

          m_t = self.beta1 * m_prev + (1 - self.beta1) * decay
          v_t = self.beta2 * v_prev + (1 - self.beta2) * (decay**2)

          self.m[param] = m_t
          self.v[param] = v_t

          m_hat = m_t / (1 - self.beta1**self.t)
          v_hat = v_t / (1 - self.beta2**self.t)

          update_term = self.lr * m_hat / (v_hat**0.5 + self.eps)
          param.data = ndl.Tensor(param.data - update_term, dtype=param.dtype)
        ### END YOUR SOLUTION
