import torch
from torch import nn
import torchvision
from torchvision import transforms
import numpy as np

net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2),
    nn.ReLU(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5),
    nn.ReLU(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120),
    nn.ReLU(),
    nn.Linear(120, 84),
    nn.ReLU(),
    nn.Linear(84, 10),
)

'''
Consider:
- Changing to MaxPool2d
- Changing number of channels
- Changing kernel sizes
- Changing number of hidden units in the fully connected layers
- Changing learning rate, batch size, number of epochs
- Adding dropout layers
'''

def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)

batch_size = 256


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
    
    net.apply(init_weights)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)

    def accuracy(hat_y, real_y):
        if len(hat_y.shape) > 1:
            hat_y = hat_y.argmax(axis=1)
        return (hat_y == real_y).sum().item()

    for ep in range(num_epochs):
        net.train()
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_hat = net(x_batch)
            loss = loss_fn(y_hat, y_batch)
            loss.backward()
            optimizer.step()
        net.eval()
        with torch.no_grad():
            metric = Accumulator(2)
            for x_batch, y_batch in train_loader:
                y_hat = net(x_batch)
                metric.add(accuracy(y_hat, y_batch), y_batch.numel())
            print(f"epoch {ep + 1}, accuracy {metric[0] / metric[1]:.4f}")
    with torch.no_grad():
        total_acc = 0
        total_num = 0
        for x_batch, y_batch in test_loader:
            y_hat = net(x_batch)
            total_acc += accuracy(y_hat, y_batch)
            total_num += y_batch.shape[0]
        print(f"test accuracy {total_acc / total_num:.4f}")


if __name__ == "__main__":
    train_and_evaluate()
