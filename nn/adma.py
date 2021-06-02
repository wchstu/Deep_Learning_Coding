import numpy as np
from .optim import Optimizer

class Adam(Optimizer):
    def __init__(self, model, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0):
        super(Adam, self).__init__(model, lr, weight_decay)
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps

        self.t = 0
        self.m = 0
        self.v = 0

    def _step(self):
        grad = []
        for module in self.model.modules.values():
            if hasattr(module, 'grad'):
                grad.append(module.grad['grad_w'])
                grad.append(None if module.grad['grad_b'] is None else module.grad['grad_b'])
        grad = np.array(grad, dtype=np.object)

        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta1) * grad ** 2
        mt = self.m / (1 - self.beta1 ** self.t)
        vt = self.v / (1 - self.beta2 ** self.t)

        return -self.lr * mt / (vt ** 0.5 + self.eps)