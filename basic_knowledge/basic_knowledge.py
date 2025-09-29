import torch

# x = torch.arange(12).reshape((3,4))
# # print(x)
# # print(x.shape)

# # torch.zeros((2,3,4))
# # torch.ones((2,3,4))
# # torch.randn(3,4)
# y = torch.tensor([[2,1,4,3],[1,2,3,4],[4,3,2,1]])
# print(torch.cat((x,y),dim = 0))
# print(torch.cat((x,y),dim = 1))
# print(x==y)

# a = torch.arange(3).reshape((3,1))
# b = torch.arange(2).reshape((1,2))
# print(a*b)

# x = torch.arange(12).reshape((3,4))
# x[0:2,:] = 12
# print(x)
y = torch.arange(2)
before = id(y)
x = torch.arange(2)
y = x+y
z = torch.zeros_like(y)
z[:] = x+y
print(id(z)==before)