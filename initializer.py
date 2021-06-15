import numpy as np
from dataset import mnist
import matplotlib.pyplot as plt
np.random.seed(10)

lr = 0.001

def cal_fan(tensor):
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
    return 2.0 / float(fan_in + fan_out)

def xaiver_uniform(tensor):
    bound = np.sqrt(cal_fan(tensor) * 3)
    return np.random.uniform(-bound, bound, tensor.shape)

class Linear:
    def __init__(self, in_features, out_features, bias=True, use_xaiver=True):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = np.random.uniform(-0.01, 0.01, (in_features, out_features))
        if use_xaiver:
            self.weight = xaiver_uniform(self.weight)
        self.grad_w = np.zeros_like(self.weight)

        self.bias = None
        self.grad_b = None
        if bias:
            self.bias = np.random.uniform(-0.01, 0.01, (1, out_features))
            if use_xaiver:
                fan = cal_fan(self.weight)
                bound = 1 / np.sqrt(fan)
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
        return self.outputs # * 4 - 2

    def backward(self, grad):
        return grad * self.outputs * (1.0 - self.outputs)   # * 4

class trans_sigmoid(Sigmoid):
    def forward(self, inputs):
        self.outputs = super().forward(inputs)
        return self.outputs * 4 - 2

    def backward(self, grad):
        return super().backward(grad) * 4

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

def model_with_trans_sogmoid(activation_fun):
    use_xavier = False
    layers = [Linear(784, 200, use_xaiver=use_xavier),
              activation_fun(),
              Linear(200, 10, use_xaiver=use_xavier)]

    loss_fun = CrossEntropyLoss()

    epochs = 5
    batch_size = 32
    train_set, valid_set, test_set = mnist('./data/mnist.pkl.gz', one_hot=True)
    X_train = train_set[0]
    Y_train = train_set[1]

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
            delta = loss.backward()
            for fun in layers[::-1]:
                delta = fun.backward(delta)
                if hasattr(fun, 'step'):
                    fun.step()

            if (i + 1) % 10 == 0:
                val_x = valid_set[0]
                for fun in layers:
                    val_x = fun.forward(val_x)
                prev = np.argmax(val_x, axis=1)
                target = np.argmax(valid_set[1], axis=1)
                acc = sum(prev == target) / len(target)
                val_acc.append(acc)
                print('epoch {}, step {}, loss = {}, val acc = {}'.format(epoch + 1, i + 1, loss, acc))
    return val_acc

if __name__ == '__main__':
    # val_acc1 = model_with_trans_sogmoid(Sigmoid)
    # val_acc2 = model_with_trans_sogmoid(trans_sigmoid)

    # plt.plot(range(1, len(val_acc1)+1), val_acc1, 'b')
    # plt.plot(range(1, len(val_acc2)+1), val_acc2, 'g')
    # plt.legend(['sigmoid', "trans_sigmoid"])
    # plt.show()


    use_xavier = True
    layers = [Linear(784, 200, use_xaiver=use_xavier),
              ReLU(),
              Linear(200, 100, use_xaiver=use_xavier),
              ReLU(),
              Linear(100, 50, use_xaiver=use_xavier),
              ReLU(),
              Linear(50, 20, use_xaiver=use_xavier),
              ReLU(),
              Linear(20, 10, use_xaiver=use_xavier)]

    loss_fun = CrossEntropyLoss()

    epochs = 10
    batch_size = 32
    train_set, valid_set, test_set = mnist('./data/mnist.pkl.gz', one_hot=True)
    X_train = train_set[0]
    Y_train = train_set[1]

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
            delta = loss.backward()
            for fun in layers[::-1]:
                delta = fun.backward(delta)
                if hasattr(fun, 'step'):
                    fun.step()

            if (i + 1) % 100 == 0:
                val_x = valid_set[0]
                for fun in layers:
                    val_x = fun.forward(val_x)
                prev = np.argmax(val_x, axis=1)
                target = np.argmax(valid_set[1], axis=1)
                acc = sum(prev == target) / len(target)
                val_acc.append(acc)
                print('epoch {}, step {}, loss = {}, val acc = {}'.format(epoch + 1, i + 1, loss, acc))

    plt.plot(range(1, len(val_acc)+1), val_acc)
    plt.show()