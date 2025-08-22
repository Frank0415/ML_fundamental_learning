import torch
import torchvision
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import time

num_inputs = 784
num_outputs = 10

W = np.random.normal(0, 0.01, (num_inputs, num_outputs))
b = np.zeros(num_outputs)


class Accumulator:  # @save
    """在n个变量上累加"""

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def train_and_evaluate(batch_size=256, lr=0.1, num_epochs=5):
    # 数据预处理
    trans = transforms.ToTensor()
    mnist_train = torchvision.datasets.FashionMNIST(
        root="./data", train=True, transform=trans, download=True
    )
    mnist_test = torchvision.datasets.FashionMNIST(
        root="./data", train=False, transform=trans, download=True
    )
    train_loader = torch.utils.data.DataLoader(
        mnist_train, batch_size=batch_size, shuffle=True, num_workers=4
    )
    test_loader = torch.utils.data.DataLoader(
        mnist_test, batch_size=batch_size, shuffle=False, num_workers=4
    )
    num_inputs = 784
    num_outputs = 10

    W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
    b = torch.zeros(num_outputs, requires_grad=True)

    def softmax(X):
        X_max = X.max(dim=1, keepdim=True).values
        X_stable = X - X_max
        X_exp = torch.exp(X_stable)
        X_sum = X_exp.sum(dim=1, keepdim=True)
        return X_exp / X_sum

    def net(X):
        X = X.reshape(-1, 784)
        return torch.matmul(X, W) + b

    def cross_entropy(logits, y):
        X_max = logits.max(dim=1, keepdim=True).values
        X_stable = logits - X_max
        log_sum_exp = torch.log(torch.exp(X_stable).sum(dim=1)) + X_max.squeeze(1)
        return log_sum_exp - logits[range(len(logits)), y]

    def accuracy(y_hat, y):
        if len(y_hat.shape) > 1 and (y_hat.shape[1] > 1):
            y_hat = y_hat.argmax(axis=1)
        cmp = y_hat.type(y.dtype) == y
        return float(cmp.type(y.dtype).sum())

    for epoch in range(num_epochs):
        for X_batch, y_batch in train_loader:
            y_hat = net(X_batch)
            l = cross_entropy(y_hat, y_batch).mean()
            l.backward()
            with torch.no_grad():
                W -= lr * W.grad
                b -= lr * b.grad
                W.grad.zero_()
                b.grad.zero_()
        # 每个epoch输出训练集准确率
        with torch.no_grad():
            total_acc = 0
            total_num = 0
            for X_batch, y_batch in train_loader:
                total_acc += accuracy(net(X_batch), y_batch)
                total_num += y_batch.shape[0]
            train_acc = total_acc / total_num
            print(f"epoch {epoch + 1}, train acc {train_acc:.4f}")
    # 测试集准确率
    with torch.no_grad():
        total_acc = 0
        total_num = 0
        for X_batch, y_batch in test_loader:
            total_acc += accuracy(net(X_batch), y_batch)
            total_num += y_batch.shape[0]
        test_acc = total_acc / total_num
        print(f"test acc {test_acc:.4f}")


if __name__ == "__main__":
    train_and_evaluate()
