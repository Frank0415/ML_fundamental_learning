import torch

x = torch.arange(4.0)
print(x)

x.requires_grad_(True)  # 等价于x=torch.arange(4.0,requires_grad=True)
x.grad

y = 2 * torch.dot(x, x)
y

y.backward()
print(x.grad == 4*x)

x.grad.zero_()
y = x.sum()
y.backward()
print(x.grad)

def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

a = torch.randn(size=(), requires_grad=True)
print(a)
d = f(a)
d.backward()
assert(a.grad == d / a)
