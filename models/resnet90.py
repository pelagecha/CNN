import torch
import torch.nn as nn


class block(nn.Module):
    def __init__(self, in_channels, out_channels, identity_donwsample=None, stride=1):
        super(block, self).__init__()
        self.expansion = 4

        # First conv layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels*out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)

        self.relu = nn.ReLU()
        self.identity_downsample = identity_donwsample

    def forward(self, x):
        identity = x
        x = self.relu(self.bn1(self.conv1(x))) # conv -> batchnorm -> relu
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))

        if self.identity_downsample:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x
    

class Model(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes) -> None:
        super(Model, self).__init__()