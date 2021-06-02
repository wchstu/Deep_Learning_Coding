import numpy as np
from dataset import mnist

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def ReLU(z):
    return np.maximum(z, 0.0)

def softmax(z):
    z_temp = np.exp(z)
    return z_temp / np.sum(z_temp, axis=1, keepdims=True)

class MLP:
    eps = 1e-8
    def __init__(self, size=(784, 100, 10), lr=0.01, lr_ratio=0.9):
        self.lr = lr
        self.lr_ratio = lr_ratio
        self.weights = []
        self.bias = []
        assert isinstance(size, list) or isinstance(size, tuple), 'size must be a list or tuple.'
        assert len(size) > 1, 'the length of size must be greater than 1.'

        for i in range(1, len(size)):
            self.weights.append(np.random.normal(0, 0.01, (size[i-1], size[i])))
            self.bias.append(np.random.normal(0, 0.01, (1, size[i])))

    def __call__(self, X, Y):
        self.target = Y

        self.z = []
        self.a = [X]
        for w, b in zip(self.weights, self.bias):
            self.z.append(self.a[-1] @ w + b)
            self.a.append(ReLU(self.z[-1]))
        self.a[-1] = softmax(self.z[-1])

        loss = -np.sum(Y * np.log(self.a[-1])) / len(Y)

        return loss

    def backward(self):
        W_grad = []
        b_grad = []
        delta = (self.a[-1] - self.target) / len(self.target)
        for i in range(len(self.weights)-1, -1, -1):
            W_grad.insert(0, self.a[i].T @ delta)
            b_grad.insert(0, np.sum(delta, axis=0, keepdims=True))
            if i == 0:
                break
            delta = delta @ (self.weights[i].T) * (self.z[i-1] > 0.0) # * self.a[i] * (1.0 - self.a[i])

        # update trained parameters
        for i in range(len(self.weights)):
            self.weights[i] -= self.lr * W_grad[i]
            self.bias[i] -= self.lr * b_grad[i]

    def predict(self, X):
        a = X
        for i in range(len(self.weights)):
            z = a @ self.weights[i] + self.bias[i]
            a = ReLU(z)

        return softmax(z)

    def validate(self, X, Y):
        pred = np.argmax(self.predict(X), axis=1)
        org = np.argmax(Y, axis=1)
        acc = sum(pred == org) / len(pred)
        return acc

    def lr_step(self):
        self.lr *= self.lr_ratio


if __name__ == '__main__':
    epochs = 20
    batch_size = 32
    train_set, valid_set, test_set = mnist('./data', one_hot=True)
    X_train = train_set[0]
    Y_train = train_set[1]

    model = MLP([784, 200, 70, 10])

    for epoch in range(epochs):
        indexs = np.arange(len(X_train))
        steps = len(X_train) // batch_size
        np.random.shuffle(indexs)
        for i in range(steps):
            ind = indexs[i*batch_size:(i+1)*batch_size]
            x = X_train[ind]
            y = Y_train[ind]
            loss = model(x, y)
            model.backward()
            # print('Epoch[{}], Step[{}], loss = {}'.format(epoch, i, loss))

            if (i + 1) % 100 == 0:
                accuracy = model.validate(valid_set[0], valid_set[1])
                print('Epoch[{}/{}] \t Step[{}/{}] \t Loss = {:.6f} \t Acc = {:.3f}'.format(epoch+1, epochs, i+1, steps, loss, accuracy))
        model.lr_step()
    print('Test Accuracy = ', model.validate(test_set[0], test_set[1]))
