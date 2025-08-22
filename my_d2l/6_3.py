import torch
import numpy
import torch.nn as nn


def comp_conv2d(conv2d, x: torch.tensor):
    x = x.reshape((1, 1) + x.shape)
    Y = conv2d(x)
    return Y.reshape(Y.shape[2:])


conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1)
conv2d_1 = nn.Conv2d(1, 1, kernel_size=(5, 3), padding=(2, 1))


conv2d_str = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
