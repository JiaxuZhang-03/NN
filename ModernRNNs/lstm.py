import torch
from torch import nn
from d2l import torch as d2l

batch_size,num_steps = 32,35
train_iter,vocab = d2l.load_data_voc(batch_size=batch_size,num_steps = num_steps)

