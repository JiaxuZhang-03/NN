import torch
from torch import nn

def corr2d(X,K):
    h,w = K.shape
    Y = torch.zeros((X.shape[0]-h+1,X.shape[1]-w+1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j] = (X[i:i+h,j:j+w]*K).sum()
    return Y
    
X = torch.ones((6,8))
X[:,2:6] = 0
K = torch.tensor([[1.0,-1.0]])
Y = corr2d(X, K)

conv2d = nn.Conv2d(1,1,kernel_size=(1,2),bias = False)

X = X.reshape((1,1,6,8))
Y = Y.reshape((1,1,6,7))
lr = 3e-2 # learning rate

# for i in range(50):
#     Y_hat = conv2d(X)
#     l = (Y_hat-Y)**2
#     conv2d.zero_grad()
#     l.sum().backward()
#     conv2d.weight.data[:] -= lr*conv2d.weight.grad
#     if (i+1)%2 == 0:
#         print(f'epoch{i+1},loss{l.sum():.3f}')
# print(conv2d.weight.data.reshape((1,2)))

def comp_conv2d(conv2d,X):
    X = X.reshape((1,1)+X.shape)
    Y = conv2d(X)
    return Y.reshape(Y.shape[2:])

conv2d = nn.Conv2d(1,1,kernel_size=3,padding = 1)
X = torch.rand(size = (8,8))
print(comp_conv2d(conv2d,X))