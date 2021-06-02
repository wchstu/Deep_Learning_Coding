import numpy as np
from .module import Module

class ReLU(Module):
    def __init__(self):
        super(ReLU, self).__init__()
        self.inputs = None

    def forward(self, inputs):
        self.inputs = inputs
        return np.maximum(inputs, 0.0)

    def backward(self, grad=None):
        return grad * (self.inputs > 0.0)

class Sigmoid(Module):
    def __init__(self):
        super(Sigmoid, self).__init__()
        self.outputs = None

    def forward(self, inputs):
        self.outputs = 1.0 / (1.0 + np.exp(-inputs))
        return self.outputs

    def backward(self, grad=None):
        return grad * self.outputs * (1.0 - self.outputs)

def softmax(z):
    z_temp = np.exp(z)
    return z_temp / np.sum(z_temp, axis=1, keepdims=True)
