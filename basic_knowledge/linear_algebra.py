import torch

# A = torch.arange(20,dtype = torch.float32).reshape(5,4)
# B = torch.arange(20).reshape(5,4)
# print(A*B+2)

# 2.3.6
# print(A.sum(axis = 1))
# print(A.sum([0,1]))
# print(A.mean(axis=0))
# sum_A = A.sum(axis = 1,keepdims = True)
# print(A/sum_A)

# print(A.cumsum(axis = 0))

# y = torch.ones(4,dtype = torch.float32)
# x = torch.arange(4,dtype = torch.float32)
# print(torch.dot(x,y))
# print(torch.sum(x*y))

# A = torch.arange(20,dtype = torch.float32).reshape(5,4)
# # x = torch.ones(4,dtype=torch.float32)
# # print(torch.mv(A,x))
# # print(torch.mv(A,x).shape)

# B = torch.ones(20,dtype = torch.float32).reshape(4,5)
# print(torch.mm(A,B))

# X = torch.tensor([1.0,2.0])
# print(torch.norm(X))
# print(torch.abs(X).sum())

# X = torch.ones((4,9))
# print(torch.norm(X))