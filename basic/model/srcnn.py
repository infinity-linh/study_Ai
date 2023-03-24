import torch.nn as nn

class SRCNN(nn.Module):
    def __init__(self, num_channels=3):
        super().__init__()

        self.conv1 = nn.Conv2d(num_channels, 64, 9, 1, 4)
        # self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        # self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 32, 5, 1, 2)
        # self.bn2 = nn.BatchNorm2d(32)

        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(32, num_channels, 5, 1, 2)
        # self.bn3 = nn.BatchNorm2d(3)

    def forward(self, x):
        # out = self.bn1(out)
        out = self.relu1(self.conv1(x))
        # out = self.bn2(out)
        out = self.relu2(self.conv2(out))
        out = self.conv3(out)
        # out = self.bn3(out)
        # out += x
        return out