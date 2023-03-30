import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, multivariate_normal as mvn
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchsummary import summary

class TwolayerMLP(nn.Module):
    def __init__(self, n, P, C):
        super(TwolayerMLP, self).__init__()
        self.input_fc = nn.Linear(n, P)
        self.output_fc = nn.Linear(P, C)
    
    def forward(self, X):
        X = self.input_fc(X)
        X = F.relu(X)
        y = self.output_fc(X)
        return y