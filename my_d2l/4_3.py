import torch
from torch import nn
import torchvision

net = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)

batch_size, epochs, lr = 256, 10, 0.1
loss_fn = nn.CrossEntropyLoss(reduction="none")
optimizer = torch.optim.SGD(net.parameters(), lr=lr)


def accuracy(y_hat, y):
    if len(y_hat.shape) > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return cmp.sum()


def train():
    transform = torchvision.transforms.ToTensor()
    mnist_train = torchvision.datasets.FashionMNIST(
        root="./data", train=True, transform=transform, download=True
    )
    train_load = torch.utils.data.DataLoader(
        mnist_train, batch_size=batch_size, shuffle=True, num_workers=4
    )

    for epoch in range(epochs):
        for x_batch, y_batch in train_load:
            optimizer.zero_grad()
            y_net = net(x_batch)
            loss = loss_fn(y_net, y_batch).mean()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            total_acc = 0
            total_num = 0
            for x_batch, y_batch in train_load:
                y_hat = net(x_batch)
                total_acc += accuracy(net(x_batch), y_batch)
                total_num += y_batch.shape[0]
            print(f"epoch {epoch + 1}, accuracy {total_acc / total_num:.4f}")
            test()


def test():
    transform = torchvision.transforms.ToTensor()
    mnist_test = torchvision.datasets.FashionMNIST(
        root="./data", train=False, transform=transform, download=True
    )
    test_load = torch.utils.data.DataLoader(
        mnist_test, batch_size=batch_size, shuffle=False, num_workers=4
    )

    total_acc = 0
    total_num = 0
    with torch.no_grad():
        for x_batch, y_batch in test_load:
            y_hat = net(x_batch)
            total_acc += accuracy(y_hat, y_batch)
            total_num += y_batch.shape[0]
    print(f"test accuracy {total_acc / total_num:.4f}")


if __name__ == "__main__":
    train()
    test()
