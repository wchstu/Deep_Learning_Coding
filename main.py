import os
import pickle
import numpy as np
import nn
from dataset import mnist

model = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
                      nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
                      nn.ReLU(),
                      nn.BatchNorm2d(6),
                      nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
                      nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
                      nn.ReLU(),
                      nn.BatchNorm2d(16),
                      nn.Flatten(),
                      nn.Dropout(0.5),
                      nn.Linear(in_features=784, out_features=120),
                      nn.ReLU(),
                      nn.Linear(in_features=120, out_features=10))
model.train()
loss_fun = nn.CrossEntropyLoss(model)
optim = nn.Adam(model, lr=0.01)

epochs = 2
batch_size = 32
train_set, valid_set, test_set = mnist('./data', one_hot=True)
X_train = train_set[0].reshape(-1, 1, 28, 28)
Y_train = train_set[1]
checkpoints_path = './checkpoints'
os.makedirs(checkpoints_path, exist_ok=True)

# with open(os.path.join(checkpoints_path, 'model_1.pickle'), 'rb') as file:
#     state_dict = pickle.load(file)
# model.load_state_dict(state_dict)
# # for k, v in state_dict.items():
# #     print(k)
# model.eval()
# prev = model(test_set[0].reshape(-1, 1, 28, 28))
# prev = np.argmax(prev, axis=1)
# target = np.argmax(test_set[1], axis=1)
# print(sum(prev == target) / len(prev))
# exit()


for epoch in range(epochs):
    indexs = np.arange(len(X_train))
    steps = len(X_train) // batch_size
    np.random.shuffle(indexs)
    for i in range(steps):
        ind = indexs[i * batch_size:(i + 1) * batch_size]
        x = X_train[ind]
        target = Y_train[ind]

        optim.zero_grad()
        prev = model(x)
        loss = loss_fun(prev, target)
        loss_fun.backward()
        optim.step()

        if (i + 1) % 100 == 0:
            model.eval()
            prev = model(valid_set[0].reshape(-1, 1, 28, 28))
            prev = np.argmax(prev, axis=1)
            target = np.argmax(valid_set[1], axis=1)
            print('epoch {}, step {}, loss = {}, val acc = {}'.format(epoch + 1, i + 1, loss,
                                                                      sum(prev == target) / len(target)))
            model.train()
    model.save(os.path.join(checkpoints_path, 'model_{}.pickle'.format(epoch+1)))