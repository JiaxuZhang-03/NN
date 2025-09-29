import torch 
from torch import nn


net = nn.Sequential(nn.Linear(4,8),nn.ReLU(),nn.Linear(8,1))
X = torch.rand(size = (2,4))
# print(net[2].state_dict())
# print(net[2].weight.data)


def block1():
    return nn.Sequential(nn.Linear(4,8),nn.ReLU(),nn.Linear(8,4),nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
        net.add_module(f'block{i}',block1())
    return net

# rgnet = nn.Sequential(block2(),nn.Linear(4,1))
# print(rgnet(X))
# print(rgnet)

def init_normal(m): # Or we can initialize it as a constant.
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight,mean = 0,std = 0.01)
        nn.init.zeros_(m.bias)
net.apply(init_normal)

def my_init(m):
    if type(m) == nn.Linear:
        nn.init.uniform_(m.weight,-10,10)
        m.weight.data *= m.weight.data.abs() >=5

net.apply(my_init)

shared = nn.Linear(8,8)
net = nn.Sequential(nn.Linear(4,8),nn.ReLU(),shared,nn.ReLU(),shared,nn.ReLU(),nn.Linear(8,1))
net(X)
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0][0] = 10
print(net[2].weight.data[0] == net[4].weight.data[0])