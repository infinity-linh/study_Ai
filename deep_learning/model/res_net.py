import torch.nn as nn
import torch
from torch import Tensor
from typing import Type
import torch.optim as optim
# from .res_net import BasicBlock

class BasicBlock(nn.Module):
    def __init__(self, num_layer, in_channels,
                 out_channels, stride=1,
                 expansion=1,
                 downsample=None) -> None:
        super(BasicBlock, self).__init__()
        self.expansion = expansion
        self.downsample = downsample
        self.num_layer = num_layer

        if num_layer > 34:
            # block 1x1, 64
            self.conv1 = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            in_channels = out_channels
        # block 3x3, 64 resnet
        self.conv2 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        if num_layer > 34:
            # block 1x1, 64 resnet 50, 101, 152, ....
            self.conv3 = nn.Conv2d(
                out_channels,
                out_channels*expansion,
                kernel_size=1,
                bias=False
            )

        else:
            # block 3x3, 64 resnet 18, 34
            self.conv3 = nn.Conv2d(
                out_channels,
                out_channels*expansion,
                kernel_size=3,
                padding=1,
                bias=False
            )

        self.bn3 = nn.BatchNorm2d(out_channels*expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        indenity = x

        if self.num_layer > 34:
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
        else:
            out = self.conv2(x)

        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            indenity = self.downsample(x)
        out += indenity
        out = self.relu(out)
        return out



class Res_Net(nn.Module):
    def __init__(self,
                 img_channels: int,
                 num_layers: int,
                 block: Type[BasicBlock] = BasicBlock,
                 num_classes: int = 1000) -> None:
        super(Res_Net, self).__init__()
        self.num_layers = num_layers
        if num_layers == 18:
            layers = [2, 2, 2, 2]
            self.expansion = 1

        elif num_layers == 34:
            layers = [3, 4, 6, 3]
            self.expansion = 1

        elif num_layers == 50:
            layers = [3, 4, 6, 3]
            self.expansion = 4

        elif num_layers == 101:
            layers = [3, 4, 23, 3]
            self.expansion = 4

        elif num_layers == 152:
            layers = [3, 8, 36, 3]
            self.expansion = 4

        self.in_channels = 64
        self.conv1 = nn.Conv2d(
            in_channels=img_channels,
            out_channels=self.in_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_layer(block, 64, layers[0])
        self.layer2 = self.make_layer(
            block, 128, layers[1], stride=2)
        self.layer3 = self.make_layer(
            block, 256, layers[2], stride=2)
        self.layer4 = self.make_layer(
            block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*self.expansion, num_classes)

    def make_layer(
            self,
            blocks,
            out_channels,
            num_block,
            stride=1
    ):
        downsample = None
        # if stride != 1:
        if stride != 1 or self.expansion != 1:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels*self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels*self.expansion)
            )
        layer = []
        layer.append(
            blocks(
                self.num_layers, 
                self.in_channels, 
                out_channels, stride, 
                self.expansion, 
                downsample
            )
        )
        self.in_channels = out_channels * self.expansion

        for i in range(1, num_block):
            layer.append(blocks(
                num_layer = self.num_layers,
                in_channels = self.in_channels,
                out_channels = out_channels,
                expansion = self.expansion
            ))
        return nn.Sequential(*layer)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # print('Dimensions of the last convolutional feature map: ', x.shape)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    tensor = torch.rand([1, 3, 512, 512])

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = Res_Net(img_channels=3,
                    num_layers=152,
                    # block=BasicBlock,
                    num_classes=1000).to(device)
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
    output = model(tensor)
    print(output.shape)
    # print(output)
