import torch

def corr2d(X: torch.tensor, k: torch.tensor):
    h, w = k.shape

    new_tensor: torch.tensor = torch.zeros(
        (X.shape[0] - k.shape[0] + 1, X.shape[1] - k.shape[1] + 1)
    )

    for i in range(new_tensor.shape[0]):
        for j in range(new_tensor.shape[1]):
            new_tensor[i, j] = (X[i : i + h, j : j + w] * k).sum()

    return new_tensor


def corr2d_multi(X: torch.tensor, K: torch.tensor):
    return sum(corr2d(x, k) for x, k in zip(X, K))


def corr2d_mul_to_mul(X: torch.tensor, K: torch.tensor):
    return torch.stack([corr2d_multi(X, k) for k in K], dim=0)


def corr2d_mul_to_mul_1x1(X: torch.tensor, K: torch.tensor):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.reshape((c_i, h * w))
    K = K.reshape((c_o, c_i))
    result = torch.matmul(X, K)
    return result.reshape((c_o, h, w))
