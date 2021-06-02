import numpy as np
from collections import OrderedDict

class Module:

    def __init__(self):
        self.modules = OrderedDict()
        self.is_training = True

    def add_module(self, name, module):
        if not isinstance(module, Module):
            raise TypeError("{} is not a Module subclass".format(module.__class__.__name__))
        self.modules[name] = module

    def forward(self, inputs):
        raise NotImplementedError("Completing forward")

    def backward(self, grad=None):
        raise NotImplementedError("Completing backward")

    def zero_grad(self):
        pass

    def step(self, grad):
        pass

    def train(self):
        self.is_training = True
        for k, v in self.modules.items():
            v.is_training = True

    def eval(self):
        self.is_training = False
        for k, v in self.modules.items():
            v.is_training = False