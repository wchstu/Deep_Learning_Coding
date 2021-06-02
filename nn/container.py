import numpy as np
import pickle
from collections import OrderedDict

from .module import Module

class Sequential(Module):
    def __init__(self, *args):
        super(Sequential, self).__init__()
        for idx, module in enumerate(args):
            self.add_module(idx, module)

    def __call__(self, inputs):
        return self.forward(inputs)

    def zero_grad(self):
        for module in self.modules.values():
            module.zero_grad()

    def forward(self, inputs):
        self.batch_size = inputs.shape[0]
        output = inputs
        for idx, module in self.modules.items():
            output = module.forward(output)
        return output

    def backward(self, grad=None):
        grad_backward = grad
        for module in reversed(self.modules.values()):
            grad_backward = module.backward(grad_backward)

    def save(self, path):
        weights_dict = self.state_dict()
        with open(path, 'wb') as file:
            pickle.dump(weights_dict, file)
        print('Model weights successfully saved in {}.'.format(path))

    def state_dict(self):
        weights_dict = OrderedDict()
        for idx, module in self.modules.items():
            name = self.__class__.__name__ + '.' +str(idx)
            if isinstance(module, Sequential):
                temp_dict = module.state_dict()
                for tk, tv in temp_dict.items():
                    weights_dict[name + '.' + tk] = tv
            else:
                if hasattr(module, 'weights'):
                    weights_dict[name + '.' + module.__class__.__name__ + '.' + 'weights'] = module.weights
                if hasattr(module, 'running_variables'):
                    weights_dict[name + '.' + module.__class__.__name__ + '.' + 'running_variables'] = module.running_variables

        return weights_dict

    def load_state_dict(self, weights_dict, prefix=''):
        for idx, module in self.modules.items():
            name = prefix + self.__class__.__name__ + '.' + str(idx)
            if isinstance(module, Sequential):
                module.load_state_dict(weights_dict, prefix=name + '.')
            else:
                if hasattr(module, 'weights'):
                    module.weights = weights_dict.pop(name + '.' + module.__class__.__name__ + '.' + 'weights')
                if hasattr(module, 'running_variables'):
                    module.running_variables = weights_dict.pop(name + '.' + module.__class__.__name__ + '.' + 'running_variables')
        if len(weights_dict) == 0:
            print('Successfully loading weights.')