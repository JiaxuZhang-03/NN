import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self,**kwargs):
        super(Encoder,self).__init__(**kwargs)
    def forward(self,X,*args):
        raise NotImplementedError