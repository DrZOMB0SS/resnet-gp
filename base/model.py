import torch
from torch import nn

class ConvFixedPadding(nn.Module):
    def __init__(
        self,
        channel_input: int,
        channel_output: int,
        kernel_size: int,
        stride: int,
        bias: bool = False,
    ):
        super(ConvFixedPadding, self).__init__()
        self.conv = nn.Conv2d(
            channel_input,
            channel_output,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class BatchNormReLU(nn.Module):
    def __init__(self, channels: int, relu: bool = True):
        super(BatchNormReLU, self).__init__()
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU() if relu else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(x)
        output = self.relu(x)

        return output


class StemBlock(nn.Module):
    def __init__(self, channel_intermediate: int):
        super(StemBlock, self).__init__()
        channel_output = channel_intermediate * 2
        self.process = nn.Sequential(
            ConvFixedPadding(3, channel_intermediate, kernel_size=3, stride=2),
            BatchNormReLU(channel_intermediate),
            ConvFixedPadding(
                channel_intermediate, channel_intermediate, kernel_size=3, stride=1
            ),
            BatchNormReLU(channel_intermediate),
            ConvFixedPadding(
                channel_intermediate, channel_output, kernel_size=3, stride=1
            ),
            BatchNormReLU(channel_output),
            ConvFixedPadding(
                channel_output, channel_output, kernel_size=3, stride=2
            ),
            BatchNormReLU(channel_output),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.process(x)

        return output


class DownSample(nn.Module):
    def __init__(self, channel_input: int, channel_output: int, stride: int):
        super(DownSample, self).__init__()
        self.down_sample = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2) if stride == 2 else nn.Identity(),
            ConvFixedPadding(channel_input, channel_output, kernel_size=1, stride=1),
            BatchNormReLU(channel_output, relu=False),
        )
        # self.down_sample = nn.Sequential(
        #         nn.Conv2d(channel_input, channel_output, kernel_size=1, stride=2) if stride == 2 
        #         else nn.Conv2d(channel_input, channel_output, kernel_size=1, stride=1),
        #         BatchNormReLU(channel_output, relu=False),
        #     )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.down_sample(x)

        return output

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
class BottleNeckBlock(nn.Module):
    expansion = 4

    def __init__(
        self,
        channel_input: str,
        channel_intermediate: int,
        stride: int,
        se_ratio: float = 0.25,
    ):
        super(BottleNeckBlock, self).__init__()
        channel_output = channel_intermediate * self.expansion
        self.down_sample = DownSample(channel_input, channel_output, stride)

        self.conv1 = nn.Sequential(
            ConvFixedPadding(
                channel_input, channel_intermediate, kernel_size=1, stride=1
            ),
            BatchNormReLU(channel_intermediate),
        )
        self.conv2 = nn.Sequential(
            ConvFixedPadding(
                channel_intermediate, channel_intermediate, kernel_size=3, stride=stride
            ),
            BatchNormReLU(channel_intermediate),
        )
        self.conv3 = nn.Sequential(
            ConvFixedPadding(
                channel_intermediate, channel_output, kernel_size=1, stride=1
            ),
            BatchNormReLU(channel_output, relu=False),
        )

        self.ca = ChannelAttention(channel_output)
        self.sa = SpatialAttention()
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.down_sample(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.ca(x) * x
        x = self.sa(x) * x
        output = self.relu(shortcut + x)

        return output


class Blocks(nn.Module):
    def __init__(
        self,
        layer: int,
        channel_input: int,
        channel_intermediate: int,
        stride: int,
        se_ratio: float,
    ):
        super(Blocks, self).__init__()
        process = [
            BottleNeckBlock(channel_input, channel_intermediate, stride, se_ratio)
        ]
        for __ in range(layer - 1):
            process.append(
                BottleNeckBlock(
                    channel_intermediate * BottleNeckBlock.expansion,
                    channel_intermediate,
                    stride=1,
                    se_ratio=se_ratio,
                )
            )
        self.process = nn.Sequential(*process)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.process(x)

        return output


class ResNetRS(nn.Module):
    def __init__(self, layers, num_class):
        super(ResNetRS, self).__init__()

        self.stem = StemBlock(32)
        self.layer1 = Blocks(layers[0], 64, 64, 1, 0.25)
        self.layer2 = Blocks(layers[1], 256, 128, 2, 0.25)
        self.layer3 = Blocks(layers[2], 512, 256, 2, 0.25)
        self.layer4 = Blocks(layers[3], 1024, 512, 2, 0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.25)
        self.fc = nn.Linear(512 * 4, num_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        output = self.fc(x)

        return output