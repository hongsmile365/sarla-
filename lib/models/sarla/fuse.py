import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.models.layers.frozen_bn import FrozenBatchNorm2d


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1,
         freeze_bn=False):
    if freeze_bn:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            FrozenBatchNorm2d(out_planes),
            nn.ReLU(inplace=True))
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True))
class Fuse(nn.Module):
    def __init__(self, channel=768*2, freeze_bn=False):
        super().__init__()
        self.conv = conv(channel, channel // 2, freeze_bn=freeze_bn)

    def forward(self, x):
        x = self.conv(x)
        return x