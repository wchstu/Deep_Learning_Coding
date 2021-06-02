import numpy as np
from .activation import softmax
from .module import Module

class CrossEntropyLoss(Module):
    def __init__(self, model):
        super(CrossEntropyLoss, self).__init__()
        self.model = model
        self.target = None

    def __call__(self, prev, target):
        self.target = target
        self.predict = softmax(prev)
        loss = -1.0 / (len(target)) * np.sum(target * np.log(self.predict))
        return loss

    def forward(self, inputs):
        pass

    def backward(self, grad=None):
        grad_backward = (self.predict - self.target) / len(self.target)
        self.model.backward(grad_backward)