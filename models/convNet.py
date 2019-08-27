import torch.nn as nn

class simpleConvNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(simpleConvNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n_channels, 15, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(15, n_classes, 1, padding=0),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


