import torch
import pandas as pd

X = torch.arange(24).view(2, 3, 4)
# print(X,X.shape,len(X[1][0]))

A = torch.arange(20).reshape(5, 4)
B = A.T

# print(A,A.shape)

A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone()  # 通过分配新内存，将A的一个副本分配给B
# print(A * B, A.sum(), A.mean(), A.std())

A_sum_axis0 = A.sum(axis=0)
# print(A_sum_axis0, A_sum_axis0.shape)
A_sum_axis1 = A.sum(axis=1)
# print(A_sum_axis1, A_sum_axis1.shape)

# print(A.mean(), A.sum() / A.numel())
# print(A.sum())
# print(A.numel())

# print(A.mean(axis=0), A.sum(axis=0) / A.shape[0])

u = torch.tensor([3.0,4.0])
# print(torch.norm(u))
# print(torch.norm(u, p=1))

P = torch.arange(24).view(2, 3, 4)
print(P)
print(P.sum(axis=0),'\n', P.sum(axis=1),'\n', P.sum(axis=2))