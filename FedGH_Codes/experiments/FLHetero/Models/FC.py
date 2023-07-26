from collections import OrderedDict

import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm
import numpy as np
import math
import torch


class FC(nn.Module):
    def __init__(self, in_dim=500, out_dim=10):
        super(FC, self).__init__()

        # self.conv1 = nn.Conv2d(in_channels, n_kernels, 5)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(n_kernels, 2* n_kernels, 5)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.fc1 = nn.Linear(2* n_kernels * 5 * 5, 400)
        # self.fc2 = nn.Linear(400, 200)
        # self.fc3 = nn.Linear(200, out_dim)
        # self.fc1 = nn.Linear(2* n_kernels * 5 * 5, 2000)
        # self.fc2 = nn.Linear(2000, 500)
        self.fc3 = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = x.view(x.shape[0], -1)
        # x = F.relu(self.fc1(x))
        # o = F.relu(self.fc2(x))
        o = self.fc3(x)
        return o




