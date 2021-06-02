from.optim import Optimizer
import numpy as np

class SGD(Optimizer):
    def __init__(self, model, lr=0.001, weight_decay=0.0):
        super(SGD, self).__init__(model, lr, weight_decay)

    def _step(self):
        grad = []
        for module in self.model.modules.values():
            if hasattr(module, 'grad'):
                grad.append(module.grad['grad_w'])
                grad.append(None if module.grad['grad_b'] is None else module.grad['grad_b'])
        grad = np.array(grad, dtype=np.object)
        return -self.lr * grad