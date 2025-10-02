import torch
from torch import nn
from torch.nn import functional as F

# net = nn.Sequential(nn.Linear(20,256),nn.ReLU(),nn.Linear(256,10))
X = torch.rand(2,20)
# print(net(X))

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20,256)
        self.out = nn.Linear(256,10)
    def forward(self,x):
        return self.out(F.relu(self.hidden(X)))
    
net = MLP()
print(net(X))

# It is a class of sequential
class MySequential(nn.Module):
    def __init__(self,*args):
        for idx,module in args:
            self._modules[str(idx)] = module
    def forward(self,X):
        for block in self._modules.values():
            X = block(X)
        return X
    
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.rand_weight = torch.rand((20,20),requires_grad=False)
        self.linear = nn.Linear(20,20)
    def forward(self,X):
        X = self.linear(X)
        X = F.relu(torch.mm(X,self.rand_weight)+1)
        X = self.linear(X)
        while X.abs().sum() >1: # a sample of control flow
            X /= 2
        return X.sum()



