import numpy as np
from .module import Module

class BatchNorm2d(Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, track_running_stats=True):
        super(BatchNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.track_running_stats = track_running_stats

        self.gamma = np.ones((1, num_features, 1, 1))
        self.beta = np.zeros((1, num_features, 1, 1))
        self.grad_gamma = np.zeros_like(self.gamma)
        self.grad_beta = np.zeros_like(self.beta)

        self.running_mean = np.zeros((1, num_features, 1, 1))
        self.running_var = np.ones((1, num_features, 1, 1))

    def forward(self, inputs):
        assert len(inputs.shape) == 4 and inputs.shape[1] == self.num_features
        if not self.is_training:
            self.numerator = inputs - self.running_mean
            self.denominator = self.running_var + self.eps
            self.inputs_norm = self.numerator / self.denominator ** 0.5
            return self.inputs_norm * self.gamma + self.beta

        mean = np.mean(inputs, axis=(0, 2, 3), keepdims=True)
        var = np.var(inputs, axis=(0, 2, 3), keepdims=True)

        if self.track_running_stats:
            self.running_mean = (1.0 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1.0 - self.momentum) * self.running_var + self.momentum * var

        self.numerator = inputs - mean
        self.denominator = var + self.eps
        self.inputs_norm = self.numerator / self.denominator ** 0.5

        return self.inputs_norm * self.gamma + self.beta

    def backward(self, grad=None):
        self.grad_gamma += np.sum(grad * self.inputs_norm, axis=(0, 2, 3), keepdims=True)
        self.grad_beta += np.sum(grad, axis=(0, 2, 3), keepdims=True)

        N = np.prod(grad.shape) / self.num_features
        grad_to_hatx = grad * self.gamma
        grad_backward = grad_to_hatx / np.sqrt(self.denominator) + \
                        np.sum(-grad_to_hatx * self.numerator * self.denominator ** (-1.5), axis=(0, 2, 3), keepdims=True) * self.numerator / N + \
                        np.sum(-grad_to_hatx / self.denominator ** 0.5, axis=(0, 2, 3), keepdims=True) / N
        return grad_backward

    def zero_grad(self):
        self.grad_gamma = np.zeros_like(self.grad_gamma)
        self.grad_beta = np.zeros_like(self.grad_beta)

    def step(self, grad):
        self.gamma += grad['grad_w']
        self.beta += grad['grad_b']

    @property
    def running_variables(self):
        return {'running_mean': self.running_mean, 'running_var':self.running_var}

    @running_variables.setter
    def running_variables(self, val):
        self.running_mean = val['running_mean']
        self.running_var = val['running_var']

    @property
    def weights(self):
        return {'weight': self.gamma, 'bias': self.beta}

    @weights.setter
    def weights(self, weights):
        self.gamma = weights['weight'].reshape(1, -1, 1, 1)
        self.beta = weights['bias'].reshape(1, -1, 1, 1)

    @property
    def grad(self):
        return {'grad_w': self.grad_gamma, 'grad_b':self.grad_beta}

    @grad.setter
    def grad(self, grad):
        self.grad_gamma = grad['grad_w'].reshape(1, -1, 1, 1)
        self.grad_beta = grad['grad_b'].reshape(1, -1, 1, 1)