import numpy as np
from dataset import mnist

import torch
from torch import nn
from torch import optim

lr = 0.01

def cal_fan(tensor, mode='fan_in'):
    shape = tensor.shape
    if len(shape) < 2:
        raise ValueError('Dimension of tensor must be greater than 2.')
    if len(shape) == 2:
        fan_in, fan_out = shape[:2]
    else:
        fan_out, fan_in = shape[:2]
    receptive_field_size = 1
    if len(shape) > 2:
        receptive_field_size = np.prod(shape[2:])
    fan_in *= receptive_field_size
    fan_out *= receptive_field_size
    return fan_in if mode=='fan_in' else fan_out

def kaiming_uniform(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    fan = cal_fan(tensor, mode)

    param = 1
    if nonlinearity == 'relu':
        param = np.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if a == None:
            a = 0.01
        param = np.sqrt(2.0 / (1 + a ** 2))
    elif nonlinearity == 'tanh':
        param = 5.0 / 3
    std = param / np.sqrt(fan)
    bound = np.sqrt(3.0) * std

    return np.random.uniform(-bound, bound, size=tensor.shape)

class Conv2d:
    # [B, C, H, W]
    def __init__(self, in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True, padding_mode='constant'):
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

    def backward(self, grad):
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

        self.grad_k_trans = np.tensordot(self.input_trans, grad_trans, [(0, 1), (0, 1)])
        if self.bias is not None:
            self.grad_b = np.sum(grad_trans, axis=(0, 1)).reshape(1, -1)
        return grad_backward

    def step(self):
        self.kernel_trans -= lr * self.grad_k_trans
        if self.bias is not None:
            self.bias -= lr * self.grad_b

    def pad(self, inputs):
        padding_width = ((0, 0),
                         (0, 0),
                         (self.padding[0], self.padding[0]),
                         (self.padding[1], self.padding[1]))
        return np.pad(inputs, pad_width=padding_width, mode=self.padding_mode)

class BatchNorm2d:
    def __init__(self, num_features, eps=1e-05, momentum=0.1, track_running_stats=True):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.track_running_stats = track_running_stats

        self.gamma = np.ones((1, num_features, 1, 1))
        self.beta = np.zeros((1, num_features, 1, 1))
        self.grad_gamma = np.ones_like(self.gamma)
        self.grad_beta = np.zeros_like(self.beta)

        self.running_mean = np.zeros((1, num_features, 1, 1))
        self.running_var = np.ones((1, num_features, 1, 1))

    def forward(self, inputs):
        assert len(inputs.shape) == 4 and inputs.shape[1] == self.num_features
        mean = np.mean(inputs, axis=(0, 2, 3), keepdims=True)
        var = np.var(inputs, axis=(0, 2, 3), keepdims=True)

        if self.track_running_stats:
            self.running_mean = (1.0 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1.0 - self.momentum) * self.running_var + self.momentum * var

        self.numerator = inputs - mean
        self.denominator = var + self.eps
        self.inputs_norm = self.numerator / self.denominator ** 0.5

        return self.inputs_norm * self.gamma + self.beta

    def backward(self, grad):
        self.grad_gamma = np.sum(grad * self.inputs_norm, axis=(0, 2, 3), keepdims=True)
        self.grad_beta = np.sum(grad, axis=(0, 2, 3), keepdims=True)

        N = np.prod(grad.shape) / self.num_features
        grad_to_hatx = grad * self.gamma
        grad_backward = grad_to_hatx / np.sqrt(self.denominator) + \
                        np.sum(-grad_to_hatx * self.numerator * self.denominator ** (-1.5), axis=(0, 2, 3), keepdims=True) * self.numerator / N + \
                        np.sum(-grad_to_hatx / self.denominator ** 0.5, axis=(0, 2, 3), keepdims=True) / N
        return grad_backward

    def step(self):
        self.grad_gamma -= lr * self.grad_gamma
        self.grad_beta -= lr * self.grad_beta

class MaxPool2d:
    def __init__(self, kernel_size, stride=(1, 1), padding=(0, 0)):
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

    def backward(self, grad):
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

class Linear:
    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = kaiming_uniform(np.empty((in_features, out_features)), a=np.sqrt(5))
        self.grad_w = np.zeros_like(self.weight)

        self.bias = None
        self.grad_b = None
        if bias:
            fan_in = cal_fan(self.weight)
            bound = 1 / np.sqrt(fan_in)
            self.bias = np.random.uniform(-bound, bound, (1, out_features))
            self.grad_b = np.zeros_like(self.bias)
        self.inputs = None

    def forward(self, inputs):
        if not isinstance(inputs, np.ndarray):
            inputs = np.array(inputs)
        assert inputs.shape[-1] == self.in_features, 'dim1({}) must be equal to in_features({})'.format(inputs.shape[-1], self.in_features)
        self.inputs = inputs
        output = inputs @ self.weight
        if self.bias is not None:
            output += self.bias
        return output

    def backward(self, grad):
        self.grad_w = self.inputs.T @ grad
        if self.bias is not None:
            self.grad_b = np.sum(grad, axis=0, keepdims=True)

        return grad @ self.weight.T

    def step(self):
        self.weight -= lr * self.grad_w
        if self.bias is not None:
            self.bias -= lr * self.grad_b

class Flatten:
    def __init__(self):
        self.shape = None

    def forward(self, inputs):
        self.shape = inputs.shape
        return inputs.reshape(self.shape[0], -1)

    def backward(self, grad):
        return grad.reshape(self.shape)

class ReLU:
    def __init__(self):
        self.inputs = None

    def forward(self, inputs):
        self.inputs = inputs
        return np.maximum(inputs, 0.0)

    def backward(self, grad):
        return grad * (self.inputs > 0.0)

class Sigmoid:
    def __init__(self):
        self.outputs = None

    def forward(self, inputs):
        self.outputs = 1.0 / (1.0 + np.exp(-inputs))
        return self.outputs

    def backward(self, grad):
        return grad * self.outputs * (1.0 - self.outputs)

def softmax(z):
    z_temp = np.exp(z)
    return z_temp / np.sum(z_temp, axis=1, keepdims=True)

class CrossEntropyLoss:
    def __init__(self):
        self.target = None
        self.prev = None
        self.loss = 0.0

    def __call__(self, prev, target):
        self.target = target
        self.predict = softmax(prev)
        self.loss = -1.0 / (len(target)) * np.sum(target * np.log(self.predict))
        return self

    def backward(self):
        return (self.predict - self.target) / len(self.target)

    def __repr__(self):
        return str(self.loss)

def ex_bp():
    layers = [Linear(in_features=784, out_features=256),
              ReLU(),
              Linear(in_features=256, out_features=128),
              ReLU(),
              Linear(in_features=128, out_features=10)]
    loss_fun = CrossEntropyLoss()

    epochs = 20
    batch_size = 32
    train_set, valid_set, test_set = mnist('./data', one_hot=True)
    X_train = train_set[0]
    Y_train = train_set[1]
    for epoch in range(epochs):
        indexs = np.arange(len(X_train))
        steps = len(X_train) // batch_size
        np.random.shuffle(indexs)
        for i in range(steps):
            ind = indexs[i * batch_size:(i + 1) * batch_size]
            x = X_train[ind]
            y = Y_train[ind]

            for fun in layers:
                x = fun.forward(x)
                # print(y.shape)
            loss = loss_fun(x, y)
            # print(i, loss)

            delta = loss.backward()
            # print(delta.shape)
            for fun in layers[::-1]:
                delta = fun.backward(delta)
                # print(delta.shape)
                if hasattr(fun, 'step'):
                    fun.step()

            if (i + 1) % 100 == 0:
                val_x = valid_set[0]
                for fun in layers:
                    val_x = fun.forward(val_x)
                prev = np.argmax(val_x, axis=1)
                target = np.argmax(valid_set[1], axis=1)
                print('epoch {}, step {}, loss = {}, val acc = {}'.format(epoch + 1, i + 1, loss,
                                                                          sum(prev == target) / len(target)))

    test_x = test_set[0]
    for fun in layers:
        test_x = fun.forward(test_x)
    prev = np.argmax(test_x, axis=1)
    target = np.argmax(test_set[1], axis=1)
    acc = sum(prev == target) / len(target)

    print("test accuracy = ", acc)

def ex_conv():
    layers = [Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
              MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
              ReLU(),
              Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
              MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
              ReLU(),
              Flatten(),
              Linear(in_features=784, out_features=120),
              ReLU(),
              Linear(in_features=120, out_features=10)]
    loss_fun = CrossEntropyLoss()

    epochs = 10
    batch_size = 32
    train_set, valid_set, test_set = mnist('./data', one_hot=True)
    X_train = train_set[0].reshape(-1, 1, 28, 28)
    Y_train = train_set[1]

    for epoch in range(epochs):
        indexs = np.arange(len(X_train))
        steps = len(X_train) // batch_size
        np.random.shuffle(indexs)
        for i in range(steps):
            ind = indexs[i * batch_size:(i + 1) * batch_size]
            x = X_train[ind]
            y = Y_train[ind]

            for fun in layers:
                x = fun.forward(x)
            loss = loss_fun(x, y)

            delta = loss.backward()
            for fun in layers[::-1]:
                delta = fun.backward(delta)
                if hasattr(fun, 'step'):
                    fun.step()

            if (i + 1) % 100 == 0:
                val_x = valid_set[0].reshape(-1, 1, 28, 28)
                for fun in layers:
                    val_x = fun.forward(val_x)
                prev = np.argmax(val_x, axis=1)
                target = np.argmax(valid_set[1], axis=1)
                print('epoch {}, step {}, loss = {}, val acc = {}'.format(epoch + 1, i + 1, loss,
                                                                          sum(prev == target) / len(target)))

    test_x = test_set[0].reshape(-1, 1, 28, 28)
    for fun in layers:
        test_x = fun.forward(test_x)
    prev = np.argmax(test_x, axis=1)
    target = np.argmax(test_set[1], axis=1)
    acc = sum(prev == target) / len(target)
    print('Test acc = ', acc)

def ex_grad():
    import torch
    from torch import nn
    conv = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    maxp = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
    relu = nn.ReLU()
    dense = nn.Linear(in_features=24, out_features=10)
    loss_torch = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD([{'params': conv.parameters()}, {'params': relu.parameters()}, {'params': dense.parameters()}], lr=0.01)

    layer = Conv2d(in_channels=3, out_channels=6, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    maxpool = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
    relu1 = ReLU()
    flatten = Flatten()
    linear = Linear(in_features=24, out_features=10)
    loss_fun = CrossEntropyLoss()

    layer.kernel = conv.weight.detach().numpy()
    layer.kernel_trans = conv.weight.detach().numpy().reshape(6, -1).T
    layer.bias = conv.bias.detach().numpy().reshape(1, -1)

    linear.weight = dense.weight.detach().numpy().T
    linear.bias = dense.bias.detach().numpy().reshape(1, -1)


    x = np.random.uniform(0, 1, (2, 3, 5, 5))
    y = np.zeros((2, 10))
    y[0, 0] = 1
    y[1, 1] = 1

    x_conv = conv.forward(torch.as_tensor(x, dtype=torch.float32))
    x_max = maxp(x_conv)
    x_relu = relu(x_max)
    x_f = torch.flatten(x_relu, start_dim=1)
    x_dense = dense(x_f)
    loss_t = loss_torch(x_dense, torch.as_tensor([0, 1]))
    print(loss_t)
    optimizer.zero_grad()
    loss_t.backward()
    for k, v in conv.named_parameters():
        if 'weight' in k:
            vn = v.grad.detach().numpy().reshape(6, -1).T


    x_layer = layer.forward(x)
    x_m = maxpool.forward(x_layer)
    x_r = relu1.forward(x_m)
    x_flatten = flatten.forward(x_r)
    x_linear = linear.forward(x_flatten)
    loss_f = loss_fun(x_linear, y)
    print(loss_f)

    grad_loss = loss_f.backward()
    grad_linear = linear.backward(grad_loss)
    grad_f = flatten.backward(grad_linear)
    grad_r = relu1.backward(grad_f)
    grad_m = maxpool.backward(grad_r)
    grad_layer = layer.backward(grad_m)

    print(layer.grad_k_trans[0])
    print(vn[0])
    print(np.sum(layer.grad_k_trans - vn))

def ex_torch():
    import torch
    from torch import nn
    conv = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
                         nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
                         nn.ReLU(),
                         nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
                         nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
                         nn.ReLU())
    linear = nn.Sequential(nn.Linear(in_features=784, out_features=120),
                           nn.ReLU(),
                           nn.Linear(in_features=120, out_features=10))

    optim = torch.optim.SGD([{'params': conv.parameters()}, {'params': linear.parameters()}], lr=0.01)
    loss_fun = nn.CrossEntropyLoss()

    epochs = 10
    batch_size = 32
    train_set, valid_set, test_set = mnist('./data', one_hot=True)
    X_train = train_set[0].reshape(-1, 1, 28, 28)
    Y_train = train_set[1]

    for epoch in range(epochs):
        indexs = np.arange(len(X_train))
        steps = len(X_train) // batch_size
        np.random.shuffle(indexs)
        for i in range(steps):
            ind = indexs[i * batch_size:(i + 1) * batch_size]
            x = X_train[ind]
            target = np.argmax(Y_train[ind], axis=1)

            x = conv(torch.as_tensor(x))
            x = torch.flatten(x, start_dim=1)
            y = linear(x)
            loss = loss_fun(y, torch.as_tensor(target))
            optim.zero_grad()
            loss.backward()
            optim.step()

            if (i + 1) % 100 == 1:
                x = valid_set[0].reshape(-1, 1, 28, 28)
                target = np.argmax(valid_set[1], axis=1)

                x = conv(torch.as_tensor(x))
                x = torch.flatten(x, start_dim=1)
                y = linear(x)
                prev = np.argmax(y.detach().numpy(), axis=1)
                print(epoch, i + 1, loss.detach(), sum(prev == target))

def ex_grad_batch():
    conv = Conv2d(3, 6, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    relu = ReLU()
    batch = BatchNorm2d(6)
    flatten = Flatten()
    linear = Linear(150, 10)
    loss_fun = CrossEntropyLoss()

    conv1 = nn.Conv2d(3, 6, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    relu1 = nn.ReLU()
    batch1 = nn.BatchNorm2d(6)
    linear1 = nn.Linear(150, 10)
    loss_fun1 = nn.CrossEntropyLoss()
    optim = optim.SGD([{'params': conv1.parameters()},
                       {'params': batch1.parameters()},
                       {'params': linear1.parameters()}], lr=0.01)

    conv.kernel_trans = conv1.weight.detach().numpy().reshape(6, -1).T
    conv.bias = conv1.bias.detach().numpy().reshape(1, -1)
    linear.weight = linear1.weight.detach().numpy().T
    linear.bias = linear1.bias.detach().numpy().reshape(1, -1)

    x = np.random.randint(0, 10, (2, 3, 5, 5))
    y = np.zeros((2, 10))
    y[0, 0] = 1
    y[1, 1] = 1
    pre = conv.forward(x)
    pre = relu.forward(pre)
    pre = batch.forward(pre)
    pre = flatten.forward(pre)
    pre = linear.forward(pre)
    loss = loss_fun(pre, y)
    grad = loss.backward()
    grad = linear.backward(grad)
    grad = flatten.backward(grad)
    grad = batch.backward(grad)
    grad = relu.backward(grad)
    grad = conv.backward(grad)
    vv = conv.grad_k_trans
    print(conv.grad_k_trans[0])

    pre1 = torch.as_tensor(x, dtype=torch.float32)
    pre1 = conv1(pre1)
    pre1 = relu1(pre1)
    pre1 = batch1(pre1)
    pre1 = torch.flatten(pre1, start_dim=1)
    pre1 = linear1(pre1)
    optim.zero_grad()
    loss1 = loss_fun1(pre1, torch.as_tensor([0, 1]))
    loss1.backward()
    for k, v in conv1.named_parameters():
        if 'weight' in k:
            vvv = v.grad.detach().numpy().reshape(6, -1).T
            print(v.grad.detach().numpy().reshape(6, -1).T[0])


if __name__ == '__main__':
    layers = [Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
              MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
              ReLU(),
              Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
              MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
              ReLU(),
              Flatten(),
              Linear(in_features=784, out_features=120),
              ReLU(),
              Linear(in_features=120, out_features=10)]
    loss_fun = CrossEntropyLoss()

    epochs = 5
    batch_size = 32
    train_set, valid_set, test_set = mnist('./data/mnist.pkl.gz', one_hot=True)
    X_train = train_set[0].reshape(-1, 1, 28, 28)
    Y_train = train_set[1]

    train_loss = []
    val_acc = []
    for epoch in range(epochs):
        indexs = np.arange(len(X_train))
        steps = len(X_train) // batch_size
        np.random.shuffle(indexs)
        for i in range(steps):
            ind = indexs[i * batch_size:(i + 1) * batch_size]
            x = X_train[ind]
            y = Y_train[ind]

            for fun in layers:
                x = fun.forward(x)
            loss = loss_fun(x, y)
            train_loss.append(float(loss.loss))
            delta = loss.backward()
            for fun in layers[::-1]:
                delta = fun.backward(delta)
                if hasattr(fun, 'step'):
                    fun.step()

            if (i + 1) % 100 == 0:
                val_x = valid_set[0].reshape(-1, 1, 28, 28)
                for fun in layers:
                    val_x = fun.forward(val_x)
                prev = np.argmax(val_x, axis=1)
                target = np.argmax(valid_set[1], axis=1)
                acc = sum(prev == target) / len(target)
                val_acc.append(acc)
                print('epoch {}, step {}, loss = {}, val acc = {}'.format(epoch + 1, i + 1, loss, acc))

    test_x = test_set[0].reshape(-1, 1, 28, 28)
    for fun in layers:
        test_x = fun.forward(test_x)
    prev = np.argmax(test_x, axis=1)
    target = np.argmax(test_set[1], axis=1)
    acc = sum(prev == target) / len(target)
    print('Test acc = ', acc)

    import matplotlib.pyplot as plt
    plt.plot(range(1, len(train_loss)+1), train_loss)
    plt.title('Training Loss')
    plt.show()
    plt.plot(range(1, len(val_acc)+1), val_acc)
    plt.title('Validation Accuracy')
    plt.show()