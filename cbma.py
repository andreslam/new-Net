import torch
import torch.nn as nn
import torch.nn.functional as F

Copy code

class CBMAAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CBMAAttention, self).__init__()
        # 定义自适应平均池化层和自适应最大池化层
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # 定义两个卷积层
        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)
        # 定义Sigmoid激活函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 对输入张量进行平均池化和最大池化操作，并分别通过self.conv1和self.conv2进行卷积操作，得到平均池化和最大池化的特征向量
        avg_out = self.conv2(F.relu(self.conv1(self.avg_pool(x))))
        max_out = self.conv2(F.relu(self.conv1(self.max_pool(x))))
        # 将平均池化和最大池化的特征向量相加，并通过Sigmoid激活函数将结果映射到0到1之间
        out = avg_out + max_out
        out = self.sigmoid(out) * x
        # 将输入张量与映射后的特征向量相乘，得到输出张量
#         return out
#     定义了一个名为CBMAAttention的类，用于实现通道注意力机制。通道注意力机制是一种用于增强模型对不同通道特征的关注度的技术，可以提高模型的性能。

# 在类的初始化函数中，定义了一个自适应平均池化层nn.AdaptiveAvgPool2d(1)和一个自适应最大池化层nn.AdaptiveMaxPool2d(1)，用于对输入张量进行池化操作。然后，定义了两个卷积层self.conv1和self.conv2，用于对池化后的特征进行卷积操作。其中，self.conv1的输入通道数为in_channels，输出通道数为in_channels // reduction，卷积核大小为1x1；self.conv2的输入通道数为in_channels // reduction，输出通道数为in_channels，卷积核大小为1x1。最后，定义了一个Sigmoid激活函数nn.Sigmoid()，用于将输出值映射到0到1之间。

# 在类的前向传播函数中，首先对输入张量进行平均池化和最大池化操作，并分别通过self.conv1和self.conv2进行卷积操作，得到平均池化和最大池化的特征向量。然后，将平均池化和最大池化的特征向量相加，并通过Sigmoid激活函数将结果映射到0到1之间。最后，将输入张量与映射后的特征向量相乘，得到输出张量。
