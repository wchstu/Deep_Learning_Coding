import numpy as np
from .module import Module
from .init import kaiming_uniform, cal_fan

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = kaiming_uniform(np.empty((in_features, out_features)), a=np.sqrt(5))
        self.grad_w = np.zeros_like(self.weight)

        self.bias = None
        self.grad_b = None
        if bias:
            fan_in = cal_fan(self.weight)
            bound = 1 / np.sqrt(fan_in)
            self.bias = np.random.uniform(-bound, bound, (1, out_features))
            self.grad_b = np.zeros_like(self.bias)
        self.inputs = None

    def forward(self, inputs):
        if not isinstance(inputs, np.ndarray):
            inputs = np.array(inputs)
        assert inputs.shape[-1] == self.in_features, 'dim1({}) must be equal to in_features({})'.format(inputs.shape[-1], self.in_features)
        self.inputs = inputs
        output = inputs @ self.weight
        if self.bias is not None:
            output += self.bias
        return output

    def backward(self, grad=None):
        self.grad_w += self.inputs.T @ grad
        if self.bias is not None:
            self.grad_b += np.sum(grad, axis=0, keepdims=True)

        return grad @ self.weight.T

    def zero_grad(self):
        self.grad_w = np.zeros_like(self.grad_w)
        if self.bias is not None:
            self.grad_b = np.zeros_like(self.grad_b)

    def step(self, grad):
        self.weight += grad['grad_w']
        if self.bias is not None:
            self.bias += grad['grad_b']

    @property
    def weights(self):
        return {'weight': self.weight, 'bias': self.bias}

    @weights.setter
    def weights(self, weights):
        self.weight = weights['weight']
        self.bias = weights['bias'].reshape(1, -1)

    @property
    def grad(self):
        return {'grad_w': self.grad_w, 'grad_b': self.grad_b}

    @grad.setter
    def grad(self, grad):
        self.grad_w = grad['grad_w']
        self.grad_b = grad['grad_b']