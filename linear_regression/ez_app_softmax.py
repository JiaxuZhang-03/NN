import torch
from torch import nn
from data_generation import load_data_fashion_mnist
from softmax_regression import train_ch3
batch_size = 256
train_iter,test_iter = load_data_fashion_mnist(batch_size)

net = nn.Sequential(nn.Flatten(),nn.Linear(784,10))

def init_weight(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight,std = 0.01)
net.apply(init_weight)

loss = nn.CrossEntropyLoss(reduction = 'none')
trainer = torch.optim.SGD(net.parameters(),lr = 0.1)
num_epochs = 10
train_ch3(net,train_iter,test_iter,loss,num_epochs,trainer)


