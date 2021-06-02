import numpy as np
from .module import Module

class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=(1, 1), padding=(0, 0)):
        super(MaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, inputs):
        inputs = self.pad(inputs)
        self.inputs_shape = inputs.shape
        self.batch_size, C, H_in, W_in = inputs.shape
        H_out = (inputs.shape[2] - self.kernel_size[0]) // self.stride[0] + 1
        W_out = (inputs.shape[3] - self.kernel_size[1]) // self.stride[1] + 1

        max_pool = np.empty((self.batch_size, C, H_out, W_out))
        self.max_flag = np.zeros_like(max_pool, dtype=np.int)

        for ih in range(H_out):
            begin_h = ih * self.stride[0]
            for iw in range(W_out):
                begin_w = iw * self.stride[1]
                temp = inputs[:, :, begin_h:(begin_h+self.kernel_size[0]), begin_w:(begin_w+self.kernel_size[1])].reshape(self.batch_size, C, -1)
                max_index = np.argmax(temp, axis=2)
                self.max_flag[:, :, ih, iw] = max_index
                max_val = np.take_along_axis(temp, max_index[:, :, np.newaxis], axis=2)
                max_pool[:, :, ih, iw] = max_val.squeeze()

        return max_pool

    def backward(self, grad=None):
        grad_backward = np.zeros(self.inputs_shape)
        for ih in range(grad.shape[2]):
            begin_h = ih * self.stride[0]
            for iw in range(grad.shape[3]):
                begin_w = iw * self.stride[1]
                max_index = self.max_flag[:, :, ih, iw]
                temp = np.zeros((self.batch_size*grad.shape[1], self.kernel_size[0]*self.kernel_size[1]))
                temp[np.arange(len(temp)), max_index.reshape(-1)] = 1
                temp = temp.reshape(self.batch_size, grad.shape[1], self.kernel_size[0], self.kernel_size[1])
                temp *= grad[:, :, ih, iw][:, :, np.newaxis, np.newaxis]

                grad_backward[:, :, begin_h:(begin_h+self.kernel_size[0]), begin_w:(begin_w+self.kernel_size[1])] += temp

        return grad_backward[:, :, self.padding[0]:self.inputs_shape[2]-self.padding[0], self.padding[1]:self.inputs_shape[3]-self.padding[1]]

    def pad(self, inputs):
        padding_width = ((0, 0),
                         (0, 0),
                         (self.padding[0], self.padding[0]),
                         (self.padding[1], self.padding[1]))
        return np.pad(inputs, padding_width, mode='constant')