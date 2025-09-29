import math
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
class Timer:
    def __init__(self):
        self.times = []
        self.start()
    def start(self):
        self.tik = time.time()
    def stop(self):
        self.times.append(time.time()-self.tik)
        return self.times[-1]
    def avg(self):
        return sum(self.times)/len(self.times)
    def sum(self):
        return sum(self.times)
    def cumsum(self):
        return np.array(self.times).cumsum().tolist()
# n = 10000
# a = torch.ones([n])
# b = torch.ones([n])
# c = torch.zeros(n)
# timer = Timer()
# timer.start()
# d = a+b
# print(f'{timer.stop():.5f} sec')

def normal(x,mu,sigma):
    p = (1/math.sqrt(2*math.pi*sigma**2))
    return p*np.exp(-0.5/sigma**2 * (x-mu)**2)

# x = np.arange(-1,1,0.1)
# params = [0,1]
# print(normal(x,params[0],params[1]))
np.random.seed(42)
X = np.linspace(0,10,50)
y = 3*X +2 + np.random.randn(50)*2

X_b = np.c_[np.ones((50,1)),X]
theta_best = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
X_new = np.linspace(0,10,100)
X_new_b = np.c_[np.ones((100,1)),X_new]
y_pred = X_new_b @ theta_best
print(y_pred)


