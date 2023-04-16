import torch
import torch.nn as nn
import torch.nn.functional as F

class CBMAAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CBMAAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.conv2(F.relu(self.conv1(self.avg_pool(x))))
        max_out = self.conv2(F.relu(self.conv1(self.max_pool(x))))
        out = avg_out + max_out
        out = self.sigmoid(out) * x
        return out