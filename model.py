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
        self.padding = (self.kernel_size - 1) // 2
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
        self.n = n
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
        self.conv1 = ConvBlock(3, 32, 3, 1)
        self.conv2 = ConvBlock(32, 64, 3, 2)
        self.block1 = ResLoop(64, 32, 1)
        self.conv3 = ConvBlock(64, 128, 3, 2)
        self.block2 = ResLoop(128, 64, 2)
        self.conv4 = ConvBlock(128, 256, 3, 2)
        self.block3 = ResLoop(256, 128, 8) # cache output for concat
        self.conv5 = ConvBlock(256, 512, 3, 2)
        self.block4 = ResLoop(512, 256, 8) # cache output for concat
        self.conv6 = ConvBlock(512, 1024, 3, 2)
        self.block5 = ResLoop(1024, 512, 4)
    
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

"""
Scale Block: yolo outputs at 3 scales
including: 3 conv2d block and one output conv layer
"""
class ScaleBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels
    ):
        super().__init__()
        self.first_in = in_channels
        self.out_channels = out_channels
        self.in_channels = self.out_channels * 2 
        self.block = nn.Sequential(
            *[
                ConvBlock(self.first_in, self.out_channels, 1, 1),
                ConvBlock(self.out_channels, self.in_channels, 3, 1),
                ConvBlock(self.in_channels, self.out_channels, 1, 1),
                ConvBlock(self.out_channels, self.in_channels, 3, 1),
                ConvBlock(self.in_channels, self.out_channels, 1, 1),
                ConvBlock(self.out_channels, self.in_channels, 3, 1),
            ]
        )
        self.out = nn.Conv2d(self.in_channels, 255, 1, 1) # outputs
    
    def forward(self, x):
        x = self.block(x)
        x = self.out(x)
        return x

"""
YOLO: 3 outputs at different scale:
    order: (8, 8)->(16, 16)->(32, 32)
    out_channels = num_anchors * (5 + 80) = 255
"""
class YOLO(nn.Module):
    def __init__(self):
        super().__init__()
        self.darknet = DarknetBody()
        self.block1 = ScaleBlock(1024, 512)
        self.conv1 = ConvBlock(255, 256, 1, 1)
        self.block2 = ScaleBlock(768, 256)
        self.conv2 = ConvBlock(255, 128, 1, 1)
        self.block3 = ScaleBlock(384, 128)

    def forward(self, x):
        out = []
        x, cache = self.darknet(x)
        x = self.block1(x)
        out.append(x)
        x = self.conv1(x)
        x = F.interpolate(x, scale_factor=2)
        assert len(cache) == 2
        x = torch.cat((x, cache[1]), dim=1) # channel-first
        x = self.block2(x)
        out.append(x)
        x = self.conv2(x)
        x = F.interpolate(x, scale_factor=2)
        x = torch.cat((x, cache[0]), dim=1)
        x = self.block3(x)
        out.append(x)
        return out

if __name__ == "__main__":
    net = YOLO()
    print(net)
    X = torch.rand(1, 3, 256, 256)
    out = net(X)
    print(len(out))
    print([out_.size() for out_ in out])