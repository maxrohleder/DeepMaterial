from torch.nn import Module
import torch.nn as nn
import torch.nn.functional as F

class ConvModel(Module):
    def __init__(self, imagesize):
        super(ConvModel, self).__init__()
        self.s = imagesize
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5, stride=2, padding=2)
        self.fc1 = nn.Linear(7*7*20, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x.view(-1, 1, self.s, self.s)))
        x2 = F.relu(self.conv2(x)).view(-1, 7*7*20)
        x3 = F.relu(self.fc1(x2))
        return x3

    def _get_name(self):
        return "SimpleConv"

class simpleFC(Module):
    def __init__(self):
        super(simpleFC, self).__init__()
        self.fc1 = nn.Linear(28*28, 1)

    def forward(self, x):
        y = F.relu(self.fc1(x.view(-1, 28*28)))
        return y

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x.view(-1, 1, 28, 28)))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)