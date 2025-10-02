import torch
from torch import nn

print(torch.mps.device_count()) # on MacBook
device = torch.device("mps")