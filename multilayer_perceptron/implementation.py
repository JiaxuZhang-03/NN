import torch 
from torch import nn
from train_functions import *
# a multilayer perceptron with 1 hidden layer
num_inputs,num_outputs,num_hiddens = 784,10,256
batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)
W1 = nn.Parameter(torch.randn(num_inputs,num_hiddens,requires_grad = True)*0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens,requires_grad=True))
W2 = nn.Parameter(torch.randn(num_hiddens,num_outputs,requires_grad=True)*0.01)
b2 = nn.Parameter(torch.zeros(num_outputs,requires_grad=True))

params = [W1,b1,W2,b2]

def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X,a)

def net(X):
    X = X.reshape((-1,num_inputs))
    H = relu(X@W1 + b1)
    return (H@W2 + b2)
if False:
    loss = nn.CrossEntropyLoss(reduction = 'none')
    num_epochs = 10
    lr = 0.1
    updater = torch.optim.SGD(params,lr = lr)
    train_ch3(net,train_iter,test_iter,loss,num_epochs,updater)

# An ez implementation of MLP
net = nn.Sequential(nn.Flatten(),nn.Linear(784,256),nn.ReLU(),nn.Linear(256,10))
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight,std = 0.01)

net.apply(init_weights)
batch_size,lr,num_epochs = 256,0.1,10
loss = nn.CrossEntropyLoss(reduction = 'none')
trainer = torch.optim.SGD(net.parameters(),lr = lr)
train_ch3(net,train_iter,test_iter,loss,num_epochs,trainer)

