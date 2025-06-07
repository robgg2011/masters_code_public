import torch
import torch.nn as nn

import numpy as np


class LogitClip(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.C =  C
    
    def forward(self, x):
        x = torch.tanh(x)
        x = torch.mul(x, self.C)
        return x



def weight_init_uniform(m, c=None):
    if isinstance(m, nn.Linear):
        if c is not None:
            m.weight.data.fill_(c)
            m.bias.data.fill_(0)
        else:
            n = m.in_features
            y = 1.0/np.sqrt(n)
            m.weight.data.uniform_(-y, y)
            m.bias.data.fill_(0)


def weight_init_normal(m):
    if isinstance(m, nn.Linear):
        n = m.in_features
        y = 1.0/np.sqrt(n)
        m.weight.data.normal_(0, y)
        m.bias.data.fill_(0)