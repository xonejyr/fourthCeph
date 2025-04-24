""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False), # size not changed
            nn.BatchNorm2d(out_channels), # [N,H,W] 归一化
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), # 池化操作，维度变化为[H/2,W/2]
            DoubleConv(in_channels, out_channels) # 用DoubleConv来进行卷积特征提取，中间和输出尺度相同
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) # [H, W]=>[2H, 2W]
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2) # 中间窄，两边宽的doubleConv
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2) # [C,H,W]=>[C/2,H*2,W*2]
            self.conv = DoubleConv(in_channels, out_channels) # 中间和输出都相同的doubleConv

    def forward(self, x1, x2):
        x1 = self.up(x1) #[B,C_in,H,W]=>[B,C_out,H*2,W*2], x1的尺度要升高，
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]  # diff in height
        diffX = x2.size()[3] - x1.size()[3]  # diff in width

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2]) # 对x1进行padding，使得维度与x2的相同，上下左右分别填充如此
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1) # 在通道维度进行拼接
        return self.conv(x)




class OutConv(nn.Module):
    """
    改变通道数，不改变尺寸
    """
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
