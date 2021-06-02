import numpy as np

class Optimizer:
    def __init__(self, model, lr=0.001, weight_decay=0.0):
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay

    def zero_grad(self):
        self.model.zero_grad()

    def step(self):
        eta = -self.lr * self.weight_decay / self.model.batch_size
        grad = self._step()
        ind = 0
        for module in self.model.modules.values():
            if hasattr(module, 'grad'):
                curgrad = {'grad_w': grad[ind] + module.weights['weight'] * eta}
                ind += 1
                if module.grad['grad_b'] is None:
                    curgrad['grad_b'] = None
                else:
                    curgrad['grad_b'] = grad[ind] + module.weights['bias'] * eta
                    ind += 1
                module.step(curgrad)

    def _step(self):
        raise NotImplementedError('Must be implemented')