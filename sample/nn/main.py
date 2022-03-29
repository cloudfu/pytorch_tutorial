from telnetlib import XDISPLOC
import torch
from torch.utils import data
from torch import nn
import os, sys
# sys.path.append(os.getcwd())
# from sample.nn.nn_model.linear_softmax_number import linear_softmax_number    

# linearModel = linear_softmax_number()
# linearModel.train()

dropout = 0.5
X = torch.randn(3,3)
print(torch.mean(X))
print(X)
target = torch.randn(X.shape)
print(target)
mask = (target > dropout).float()
print(mask)
output = mask * X / (1.0 - dropout)
print(torch.mean(output))