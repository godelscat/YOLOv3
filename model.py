"""
Build Darknet53 for yolov3
Ref: https://github.com/qqwweee/keras-yolo3
"""
import torch
from torch import nn
import torch.nn.functional as F

"""
ConvBlock: including conv2d, batchNormalization, LeakyReLU
"""
class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (self.kernel_size - 1) // self.stride
        self.conv = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            padding=self.padding
        )
        self.bn = nn.BatchNorm2d(self.out_channels)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.leaky_relu(x, 0.1, inplace=True)
        return x


"""
ConvRes Block: including conv2d, batchNormalization, LeakyReLU and Residual;
"""
class ConvRes(nn.Module):
    def __init__(
        self, 
        in_channels,
        out_channels,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block1 = ConvBlock(
            self.in_channels,
            self.out_channels,
            kernel_size=1,
            stride=1
        )
        self.block2 = ConvBlock(
            self.out_channels,
            self.in_channels,
            kernel_size=3,
            stride=1
        )

    def forward(self, x):
        shortcut = x
        x = self.block1(x)
        x = self.block2(x) # not sure should cache here for concat or not
        x = torch.add(shortcut, x) 
        return x

"""
ResLoop: Residual Block Loop;
n: the number of block
"""
class ResLoop(nn.Module):
    def __init__(
        self,   
        in_channels,
        out_channels,
        n
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n
        self.block = nn.Sequential(
            *[ConvRes(self.in_channels, self.out_channels) for i in range(self.n)]
        )

    def forward(self, x):
        x = self.block(x)
        return x

"""
Darknet Body: not include the last three layers:
    Avgpool, Connected and Softmax
default input image is (3, 256, 256), channel-first
"""
class DarknetBody(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.block1 = ResLoop(64, 32, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1)
        self.block2 = ResLoop(128, 64, 2)
        self.conv4 = nn.Conv2d(128, 256, 3, 2, 1)
        self.block3 = ResLoop(256, 128, 8) # cache output for concat
        self.conv5 = nn.Conv2d(256, 512, 3, 2, 1)
        self.block4 = ResLoop(512, 256, 8) # cache output for concat
        self.conv6 = nn.Conv2d(512, 1024, 3, 2, 1)
        self.block5 = nn.Conv2d(1024, 512, 4)
    
    def forward(self, x):
        cache = []
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.block1(x)
        x = self.conv3(x)
        x = self.block2(x)
        x = self.conv4(x)
        x = self.block3(x)
        cache.append(x)
        x = self.conv5(x)
        x = self.block4(x)
        cache.append(x)
        x = self.conv6(x)
        x = self.block5(x)
        return x, cache