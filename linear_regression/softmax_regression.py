import torch
from data_generation import load_data_fashion_mnist
from easy_realization import sgd

num_inputs = 784
num_outputs = 10
batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)
W = torch.normal(0,0.01,size = (num_inputs,num_outputs),requires_grad=True)
b = torch.zeros(num_outputs,requires_grad=True)


class Accumulator:  #@save
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1,keepdim = True)
    return X_exp / partition
def net(X):
    return softmax(torch.matmul(X.reshape((-1,W.shape[0])),W) + b)

def cross_entropy(y_hat,y):
    return -torch.log(y_hat[range(len(y_hat)),y])

def accuracy(y_hat,y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis = 1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net,data_iter):
    if isinstance(net,torch.nn.Module):
        net.eval()
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X),y),y.numel())
        return metric[0]/metric[1]
    

# print(evaluate_accuracy(net, test_iter))


def train_epoch_ch3(net,train_iter,loss,updater):
    if isinstance(net,torch.nn.Module):
        net.train()
    metric = Accumulator(3)
    for X,y in train_iter:
        y_hat = net(X)
        l = loss(y_hat,y)
        if isinstance(updater,torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()),accuracy(y_hat,y),y.numel())
    return metric[0]/metric[2], metric[1]/metric[2]

def train_ch3(net,train_iter,test_iter,loss,num_epochs,updater):
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net,train_iter,loss,updater)
        test_acc = evaluate_accuracy(net,test_iter)
        train_loss, train_acc = train_metrics
        print(f"epoch {epoch+1}, loss {train_loss:.3f}, "
            f"train_acc {train_acc:.3f}, test_acc {test_acc:.3f}")
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7,train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc


def updater(batch_size,lr = 0.1):
    return sgd([W,b],lr,batch_size)

num_epochs = 10
train_ch3(net,train_iter,test_iter,cross_entropy,num_epochs,updater)
