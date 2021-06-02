
import numpy as np
from .init import kaiming_uniform, cal_fan
from .module import Module

class Conv2d(Module):
    # [B, C, H, W]
    def __init__(self, in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True, padding_mode='constant'):
        super(Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        assert padding_mode in ['constant', 'reflect'], 'padding mode must be constant or reflect'
        self.padding_mode = padding_mode

        self.kernel = kaiming_uniform(np.empty((out_channels, in_channels, kernel_size[0], kernel_size[1])), a=np.sqrt(5))
        if bias:
            fan_in = cal_fan(self.kernel)
            bound = 1 / np.sqrt(fan_in)
            self.bias = np.random.uniform(-bound, bound, (1, out_channels))
            self.grad_b = np.zeros_like(self.bias)
        else:
            self.bias = None
            self.grad_b = None
        self.kernel_trans = self.kernel.reshape(out_channels, -1).T   # shape: [in_channels*k_width*k_height, out_channels]

        self.grad_k_trans = np.zeros_like(self.kernel_trans)

    def forward(self, inputs):
        inputs = self.pad(inputs)
        self.input_shape = inputs.shape
        self.batch_size, in_channels, self.H_in, self.W_in = inputs.shape
        assert in_channels == self.in_channels, 'inputs dim1({}) is not equal to convolutional in_channels({})'.format(in_channels, self.in_channels)

        self.H_out = (inputs.shape[2] - self.kernel_size[0]) // self.stride[0] + 1
        self.W_out = (inputs.shape[3] - self.kernel_size[1]) // self.stride[1] + 1

        self.input_trans = np.empty((self.batch_size, self.H_out * self.W_out, self.kernel_trans.shape[0]))

        ind = 0
        h = 0
        while (h + self.kernel_size[0] <= inputs.shape[2]):
            w = 0
            while (w + self.kernel_size[1] <= inputs.shape[3]):
                self.input_trans[:, ind, :] = inputs[:, :, h:h + self.kernel_size[0], w:w + self.kernel_size[1]].reshape(self.batch_size, -1)
                w += self.stride[1]
                ind += 1
            h += self.stride[0]

        output = self.input_trans @ self.kernel_trans
        output = output.transpose([0, 2, 1]).reshape(self.batch_size, self.out_channels, self.H_out, self.W_out)
        if self.bias is not None:
            output += self.bias.reshape(1, -1, 1, 1)

        return output

    def backward(self, grad=None):
        grad_trans = grad.transpose([0, 2, 3, 1]).reshape(self.batch_size, -1, self.out_channels)
        grad_backward_trans = np.tensordot(grad_trans, self.kernel_trans.T, [(2), [0]])
        grad_backward = np.zeros(self.input_shape)

        ind = 0
        for ih in range(grad.shape[2]):
            begin_h = ih * self.stride[0]
            for iw in range(grad.shape[3]):
                begin_w = iw * self.stride[1]
                grad_backward[:, :, begin_h:(begin_h+self.kernel_size[0]), begin_w:(begin_w+self.kernel_size[1])] += \
                grad_backward_trans[:, ind, :].reshape(self.batch_size, self.in_channels, self.kernel_size[0], self.kernel_size[1])
                ind += 1
        grad_backward = grad_backward[:, :, self.padding[0]:self.input_shape[2]-self.padding[0], self.padding[1]:self.input_shape[3]-self.padding[1]]
        # print(grad_backward.shape)

        self.grad_k_trans += np.tensordot(self.input_trans, grad_trans, [(0, 1), (0, 1)])
        if self.bias is not None:
            self.grad_b += np.sum(grad_trans, axis=(0, 1)).reshape(1, -1)
        return grad_backward

    def step(self, grad):
        self.kernel += grad['grad_w']
        self.kernel_trans = self.kernel.reshape(self.out_channels, -1).T
        if self.bias is not None:
            self.bias += grad['grad_b']

    def pad(self, inputs):
        padding_width = ((0, 0),
                         (0, 0),
                         (self.padding[0], self.padding[0]),
                         (self.padding[1], self.padding[1]))
        return np.pad(inputs, pad_width=padding_width, mode=self.padding_mode)

    def zero_grad(self):
        self.grad_k_trans = np.zeros_like(self.grad_k_trans)
        if self.bias is not None:
            self.grad_b = np.zeros_like(self.grad_b)

    @property
    def weights(self):
        return {'weight': self.kernel_trans.T.reshape(self.kernel.shape), 'bias': self.bias}

    @weights.setter
    def weights(self, weights):
        self.kernel = weights['weight']
        self.kernel_trans = self.kernel.reshape(self.out_channels, -1).T
        self.bias = weights['bias'].reshape(1, -1)

    @property
    def grad(self):
        return {'grad_w': self.grad_k_trans.T.reshape(self.kernel.shape), 'grad_b': self.grad_b}

    @grad.setter
    def grad(self, grad):
        self.grad_k_trans = grad['grad_w'].reshape(self.out_channels, -1).T
        self.grad_b = grad['grad_b']