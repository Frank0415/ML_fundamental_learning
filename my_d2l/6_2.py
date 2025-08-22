import torch
import numpy


def corr2d(X: torch.tensor, k: torch.tensor):
    h, w = k.shape

    new_tensor: torch.tensor = torch.zeros(
        (X.shape[0] - k.shape[0] + 1, X.shape[1] - k.shape[1] + 1)
    )

    for i in range(new_tensor.shape[0]):
        for j in range(new_tensor.shape[1]):
            new_tensor[i, j] = (X[i : i + h, j : j + w] * k).sum()

    return new_tensor


def test1():
    X = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    X.to(torch.float32)
    K = torch.tensor([[0, 1], [2, 3]])
    K.to(torch.float32)

    print(corr2d(X, K))


def test2():
    X = torch.zeros((6, 8))
    X[:, 2:6] = 1
    X = X.float()
    K = torch.tensor([[1.0, -1.0]]).float()
    # print(corr2d(X,K))
    # print(corr2d(X.t(),K))
    return X, K


def test3():
    X, k = test2()
    Y = corr2d(X, k)
    X = X.reshape((1, 1, 6, 8))
    Y = Y.reshape((1, 1, 6, 7))
    for lr in [1e-2, 2e-2, 3e-2]:
        conv2d = torch.nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)
        optimizer = torch.optim.SGD(conv2d.parameters(), lr=lr)
        for i in range(20):
            Y_hat = conv2d(X)
            loss = (Y_hat - Y) ** 2
            optimizer.zero_grad()
            loss.sum().backward()
            optimizer.step()
            if (i + 1) % 4 == 0:
                print(f"epoch {i + 1}, loss {loss.sum().item():f}")
        print(conv2d.weight.data.reshape(k.shape))


if __name__ == "__main__":
    test3()
