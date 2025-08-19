import torch
import torchvision
from torch.utils import data
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import time
# 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式，
# 并除以255使得所有像素的数值均在0～1之间
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root="./data", train=True, transform=trans, download=True
)
mnist_test = torchvision.datasets.FashionMNIST(
    root="./data", train=False, transform=trans, download=True
)

print(len(mnist_train), len(mnist_test))

print(mnist_train[0][0].shape)


def get_fashion_mnist_labels(labels):  # @save
    text_labels = [
        "t-shirt",
        "trouser",
        "pullover",
        "dress",
        "coat",
        "sandal",
        "shirt",
        "sneaker",
        "bag",
        "ankle boot",
    ]
    return [text_labels[i] for i in labels]


def show_fashion_mnist(images, labels):  # @save
    """绘制图像列表"""
    figsize = (12, 12)
    _, axes = plt.subplots(5, 5, figsize=figsize)
    axes = axes.flatten()
    for i, (image, label) in enumerate(zip(images, labels)):
        axes[i].imshow(image.reshape((28, 28)), cmap="gray")
        axes[i].set_title(label)
        axes[i].axes.get_xaxis().set_visible(False)
        axes[i].axes.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.show()


def get_dataloader_workers():  # @save
    """使用4个进程来读取数据"""
    return 4


if __name__ == "__main__":
    # 取出前25个样本
    X, y = [], []
    for i in range(25):
        X.append(mnist_train[i][0])
        y.append(mnist_train[i][1])

    # 显示图像
    # show_fashion_mnist(X, get_fashion_mnist_labels(y))

    batch_size = 256
    
    start = time.clock_gettime_ns(time.CLOCK_MONOTONIC)

    train_iter = data.DataLoader(
        mnist_train, batch_size, shuffle=False, num_workers=get_dataloader_workers()
    )
    
    for X, y in train_iter:
        continue

    print("time:", (time.clock_gettime_ns(time.CLOCK_MONOTONIC) - start) / 1e9, "s")
