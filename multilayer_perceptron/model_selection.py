import math
import numpy as np
import torch
from torch import nn
from train_functions import Accumulator
max_degree = 20
n_train,n_test = 100,100
true_w = np.zeros(max_degree)
true_w[0:4] = np.array([5,1.2,-3.4,5.6])
features = np.random.normal(size = (n_train+n_test,1))
np.random.shuffle(features)
poly_features = np.power(features,np.arange(max_degree).reshape(1,-1))
for i in range(max_degree):
    poly_features[:,i] /= math.gamma(i+1)

labels = np.dot(poly_features,true_w)
labels += np.random.normal(scale=0.1,size = labels.shape)
true_w, features, poly_features, labels = [torch.tensor(x, dtype=
    torch.float32) for x in [true_w, features, poly_features, labels]]

def eval_loss(net,data_iter,loss):
    metric = Accumulator(2)
    for X,y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)
        l = loss(out,y)
        metric.add(l.sum(),l.numel())
    return metric[0]/metric[1]

def train(train_features,test_features,train_labels,test_labels,num_epochs = 100):
    loss = nn.MSELoss(reduction = 'none')
    input_shape = train_features.shape(-1)
    net = nn.Sequential(nn.Linear(input_shape,1,bias = False))
    batch_size = min(10,train_labels.shape[0])
    

