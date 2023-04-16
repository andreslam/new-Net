"""Split-Attention"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Conv2d, Module, Linear, BatchNorm2d, ReLU
from torch.nn.modules.utils import _pair

__all__ = ['SplAtConv2d']

# SplAtConv2d类，定义了split-attention卷积层
class SplAtConv2d(Module):
    """Split-Attention Conv2d"""

    # 构造函数，定义卷积层的各个参数
    def __init__(self, in_channels, channels, kernel_size, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=True, radix=2, reduction_factor=4, rectify=False, rectify_avg=False, norm_layer=None, dropblock_prob=0.0, **kwargs):
        super(SplAtConv2d, self).__init__()
        padding = _pair(padding)
        # 判断是否需要进行rectify操作
        self.rectify = rectify and (padding[0] > 0 or padding[1] > 0)
        self.rectify_avg = rectify_avg
        # 计算inter_channels的大小
        inter_channels = max(in_channels*radix//reduction_factor, 32)
        self.radix = radix
        self.cardinality = groups  # 定义群组数
        self.channels = channels
        self.dropblock_prob = dropblock_prob

        if self.rectify:
            from rfconv import RFConv2d
            # 进行rectify操作
            self.conv = RFConv2d(in_channels, channels*radix, kernel_size, stride, padding, dilation, groups=groups*radix, bias=bias, average_mode=rectify_avg, **kwargs)
        else:
            self.conv = Conv2d(in_channels, channels*radix, kernel_size, stride, padding, dilation, groups=groups*radix, bias=bias, **kwargs)

        # 是否使用BatchNorm2d
        self.use_bn = norm_layer is not None
        self.bn0 = norm_layer(channels*radix)
        self.relu = ReLU(inplace=True)  # ReLU激活函数
        self.fc1 = Conv2d(channels, inter_channels, 1, groups=self.cardinality)
        self.bn1 = norm_layer(inter_channels)
        self.fc2 = Conv2d(inter_channels, channels*radix, 1, groups=self.cardinality)

        if dropblock_prob > 0.0:
            self.dropblock = DropBlock2D(dropblock_prob, 3)

    # 前向传播
    def forward(self, x):
        x = self.conv(x)  # 先进行卷积操作
        if self.use_bn:
            x = self.bn0(x)  # 再进行BatchNorm2d操作
        if self.dropblock_prob > 0.0:
            x = self.dropblock(x)  # 再进行DropBlock操作
        x = self.relu(x)  # ReLU激活

        # 对输入的特征图进行split操作和全局平均池化
        batch, channel = x.shape[:2]  # 获取batch和channel的大小
        if self.radix > 1:
            splited = torch.split(x, channel//self.radix, dim=1)  # 对特征图进行split操作
            gap = sum(splited)  # 对split后的结果进行求和操作
        else:
            gap = x  # 仅进行全局平均池化

        gap = F.adaptive_avg_pool2d(gap, 1)  # 进行全局平均池化（adaptive avgpool2d）

        # 在通道维度上对gap执行卷积和BatchNorm2d操作
        gap = self.fc1(gap)
        if self.use_bn:
            gap = self.bn1(gap)
        gap = self.relu(gap)

        # 求取atten，然后进行softmax或sigmoid操作
        atten = self.fc2(gap).view((batch, self.radix, self.channels))
        if self.radix > 1:
            atten = F.softmax(atten, dim=1).view(batch, -1, 1, 1)
        else:
            atten = F.sigmoid(atten, dim=1).view(batch, -1, 1, 1)

        # 对split后的结果进行加权求和
        if self.radix > 1:
            atten = torch.split(atten, channel//self.radix, dim=1)
            out = sum([att*split for (att, split) in zip(atten, splited)])
        else:
            out = atten * x

        return out.contiguous()  # 将输出结果进行内存整理，返回结果