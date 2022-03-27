import torch
from torch.utils import data
from torch import nn
import os, sys
sys.path.append(os.getcwd())
from sample.nn.nn_model.linear_softmax_number import linear_softmax_number    

linearModel = linear_softmax_number()
linearModel.train()

