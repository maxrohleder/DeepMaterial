from torch.nn import Module
import torch.nn as nn
import torch.nn.functional as F

class ConvModel(Module):
    def __init__(self):
        super(ConvModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5, stride=2, padding=2)
        self.fc1 = nn.Linear(7*7*20, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x))
        x2 = x2.view(-1, 7*7*20)
        x3 = F.relu(self.fc1(x2))
        return x3
