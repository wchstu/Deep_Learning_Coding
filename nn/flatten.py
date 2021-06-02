import numpy as np
from .module import Module

class Flatten(Module):
    def __init__(self):
        super(Flatten, self).__init__()
        self.shape = None

    def forward(self, inputs):
        self.shape = inputs.shape
        return inputs.reshape(self.shape[0], -1)

    def backward(self, grad=None):
        return grad.reshape(self.shape)