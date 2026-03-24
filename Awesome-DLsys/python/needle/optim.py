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


def _nd(x):
    return x.data if hasattr(x, "data") else x


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for p in self.params:
            if p.grad is None:
                continue
            g = p.grad.detach()
            if self.weight_decay and self.weight_decay != 0.0:
                g = g + self.weight_decay * p.data.detach()
            u_prev = self.u.get(p, None)
            if u_prev is None:
                u_prev = g * 0
            u_new = self.momentum * u_prev + (1.0 - self.momentum) * g
            self.u[p] = u_new.detach()
            p.data = (p.data - self.lr * u_new).detach()

        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        Note: This does not need to be implemented for HW2 and can be skipped.
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
        for p in self.params:
            if p.grad is None:
                continue
            g = p.grad.detach()
            if self.weight_decay and self.weight_decay != 0.0:
                g = g + self.weight_decay * p.data.detach()
            m_prev = self.m.get(p, None)
            v_prev = self.v.get(p, None)
            if m_prev is None:
                m_prev = g * 0
            if v_prev is None:
                v_prev = g * 0
            m = self.beta1 * m_prev + (1.0 - self.beta1) * g
            v = self.beta2 * v_prev + (1.0 - self.beta2) * (g * g)
            self.m[p] = m.detach()
            self.v[p] = v.detach()
            m_hat = m / (1.0 - self.beta1 ** self.t)
            v_hat = v / (1.0 - self.beta2 ** self.t)
            p.data = (p.data - self.lr * m_hat / (v_hat ** 0.5 + self.eps)).detach()
        ### END YOUR SOLUTION
