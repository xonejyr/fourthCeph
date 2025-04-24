import torch
import torch.nn as nn
from collections import OrderedDict


class wblock(nn.Module):
    def __init__(self, conv3or1, in_channel, out_channel, y_channel=0):
        super(wblock, self).__init__()
        self.conv3or1 = conv3or1
        self.conv11 = nn.Sequential(OrderedDict([
            ('conv11_1', nn.Conv2d(in_channel + y_channel, out_channel, kernel_size=1)),
            ('norm11_1', nn.BatchNorm2d(out_channel)),
            ('relu11_1', nn.ReLU(inplace=True))
        ]))
        self.conv33 = nn.Sequential(OrderedDict([
            ('conv11_1', nn.Conv2d(in_channel + y_channel, out_channel, kernel_size=3, stride=1, padding=1)),
            ('norm11_1', nn.BatchNorm2d(out_channel)),
            ('relu11_1', nn.ReLU(inplace=True))
        ]))

    def forward(self, x, y):
        x = torch.cat((x, y), 1)
        if self.conv3or1 == 1:
            x = self.conv11(x)
        elif self.conv3or1 == 3:
            x = self.conv33(x)
        return x