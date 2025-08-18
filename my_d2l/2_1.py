import torch
import numpy

A = torch.arange(9).view(3, 3)
# print(A)
# print(id(A))

A = torch.zeros(10,10)
# print(A)
# print(id(A))

X = torch.tensor([1])
B = X.numpy()
print(type(X),type(B))
print(type(X.item()))
