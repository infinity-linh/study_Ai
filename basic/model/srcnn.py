import torch.nn as nn

class SRCNN(nn.Module):
    def __init__(self, num_channels=3):
        super().__init__()

        # Patch extraction and representation
        self.conv1 = nn.Conv2d(num_channels, 64, 9, 1, 4)
        self.relu1 = nn.ReLU(inplace=True)

        # Non-linear mapping
        self.conv2 = nn.Conv2d(64, 32, 5, 1, 2)
        self.relu2 = nn.ReLU(inplace=True)

        # Reconstruction
        self.conv3 = nn.Conv2d(32, num_channels, 5, 1, 2)

        # Initialize weight

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.conv3(x)

        return x