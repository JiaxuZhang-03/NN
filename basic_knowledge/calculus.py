import torch
import numpy as np

# def f(x):
#     return 3*x**2 - 4*x

# def numerical_lim(f,x,h):
#     return (f(x+h)-f(x))/h

# h = 0.1

# for i in range(5):
#     print(f'h = h{h:.5f}, numerical limit = {numerical_lim(f,1,h):.5f}')
#     h *= 0.1

# x = torch.arange(4.0,requires_grad=True)

# y = 2*torch.dot(x,x)
# y.backward()
# print(x.grad)
# x.grad.zero_()
# y = x.sum()
# y.backward()
# print(x.grad)
# x.grad.zero_()
# y = x*x
# y.sum().backward()
# print(x.grad)

# x.grad.zero_()
# y = x*x
# u = y.detach()
# z = u*x
# z.sum().backward()
# print(x.grad == u)

def f(a):
    b = a*2
    while b.norm() < 1000:
        b = b*2
    if b.sum() >0:
        c = b
    else:
        c = 100*b
    return c

a = torch.randn(size = (),requires_grad= True)
d = f(a)
d.backward()
print(a.grad == d/a)