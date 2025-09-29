import torch
from torch import nn
from torch.nn import functional as F

class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,X):
        return X-X.mean()
    
net = nn.Sequential(nn.Linear(8,128),CenteredLayer())
Y = net(torch.rand(4,8))

class MyLinear(nn.Module):
    def __init__(self,in_units,units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units,units))
        self.bias = nn.Parameter(torch.randn(units,))
    def forward(self,X):
        linear = torch.matmul(X,self.weight.data) + self.bias.data
        return F.relu(linear)

net = nn.Sequential(MyLinear(64,8),MyLinear(8,1))
print(net(torch.rand(2,64)))
