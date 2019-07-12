from torch.nn import Module
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ConvModel(Module):
    def __init__(self):
        super(ConvModel, self).__init__()
        print('init ConvModel!')
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5, stride=2, padding=2)
        self.fc1 = nn.Linear(7*7*20, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 7*7*20)
        x = F.relu(self.fc1(x))
        return x
