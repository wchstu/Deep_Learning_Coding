import numpy as np
from .module import Module

class Dropout(Module):
    def __init__(self, p=0.5):
        super(Dropout, self).__init__()
        self.p = p
        self._multiplier = None

    def forward(self, inputs):
        if self.is_training:
            self._multiplier = (np.random.uniform(size=inputs.shape) > self.p).astype(np.float)
            return inputs * self._multiplier / (1.0 - self.p)
        else:
            return inputs

    def backward(self, grad=None):
        return grad * self._multiplier / (1.0 - self.p)
