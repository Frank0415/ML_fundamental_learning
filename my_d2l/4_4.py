"""
TODO: use PyTorch to implement polynomial regression
"""

import math
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt


def calc_linreg(x: torch.tensor, w: torch.tensor, size: int):
    # x: (n,1), w: (size,1) -> return (n,1)
    # build matrix [1, x, x^2, ..., x^(size-1)] and multiply by weights
    powers = [x**i / math.factorial(i) for i in range(size)]
    X = torch.cat(powers, dim=1)  # shape (n, size)
    return X.matmul(w[:size])


def loss(x: torch.tensor, y: torch.tensor, w: torch.tensor, size: int) -> torch.tensor:
    y_hat = calc_linreg(x, w, size)
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def data_iter(x, y, batch_size):
    num_examples = x.shape[0]
    indices = torch.randperm(num_examples)
    for i in range(0, num_examples, batch_size):
        batch_indices = indices[i : i + batch_size]
        yield x[batch_indices], y[batch_indices]


def train(
    x: torch.tensor,
    y: torch.tensor,
    w: torch.tensor,
    size: int,
    lr: float = 0.01,
    batch_size: int = 10,
):
    for X_batch, y_batch in data_iter(x, y, batch_size):
        loss_net = loss(X_batch, y_batch, w, size)
        loss_net.sum().backward()
        with torch.no_grad():
            # update all relevant parameters at once using w.grad
            w[:size] -= lr * w.grad[:size] / batch_size
            w.grad.zero_()
        with torch.no_grad():
            train_loss = loss(X_batch, y_batch, w, size)
            print(f"loss {float(train_loss.mean()):f}")


real_w = torch.tensor([3, 2, 1])
real_w = torch.cat([real_w, torch.zeros(7)])
real_w = real_w.view(-1, 1)

real_x = torch.rand(100, 1) * 10 - 5
real_y = calc_linreg(real_x, real_w, 3) + torch.normal(0, 2, size=(100, 1))

# create larger test set to match reference
test_x = torch.rand(10, 1) * 12 - 6
test_y = calc_linreg(test_x, real_w, 3) + torch.normal(0, 3, size=(10, 1))

# training hyperparameters
epochs = 1000
lr = 0.01

if __name__ == "__main__":
    degrees = list(range(1, 11))
    train_mse_list = []
    test_mse_list = []

    for d in degrees:
        # reinitialize weights for this model complexity
        w = torch.zeros(size=(10, 1), requires_grad=True)
        for ep in range(epochs):
            l = loss(real_x, real_y, w, d).mean()
            l.backward()
            with torch.no_grad():
                w[:d] -= lr * w.grad[:d]
                w.grad.zero_()
        with torch.no_grad():
            train_loss = loss(real_x, real_y, w, d).mean().item()
            test_loss = loss(test_x, test_y, w, d).mean().item()
            train_mse_list.append(train_loss)
            test_mse_list.append(test_loss)
            print(
                f"degree {d} done. train loss {train_loss:.6f}, test loss {test_loss:.6f}"
            )

    # plot train/test MSE vs degree
    plt.figure(figsize=(8, 4))
    plt.plot(degrees, train_mse_list, "o-", label="train MSE")
    plt.plot(degrees, test_mse_list, "o-", label="test MSE")
    plt.xlabel("polynomial degree")
    plt.ylabel("MSE")
    plt.xticks(degrees)
    plt.legend()
    plt.title("Underfitting vs Overfitting (train/test MSE) - gradient descent")
    plt.show()
