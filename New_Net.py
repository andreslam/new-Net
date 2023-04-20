# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torchvision import models
import resnest
from splat import SplAtConv2d
from cbma import CBMAAttention

# class SELayer(nn.Module):
#     def __init__(self, channel, reduction=32):
#         super(SELayer, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(channel, channel // reduction, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(channel // reduction, channel, bias=False),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         module_input = x
#         x = self.avg_pool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x).view(x.size(0), x.size(1), 1, 1)
#         return module_input * x
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)




class res_conv_block(nn.Module):  # 定义一个名为 res_conv_block 的类，该类继承自 nn.Module。
    def __init__(self, ch_in, ch_out):  # 类的初始化函数，传入两个参数：输入通道数 ch_in 和输出通道数 ch_out。
        super(res_conv_block, self).__init__()  # 调用父类（nn.Module）的 __init__() 方法。
        self.conv = nn.Sequential(  # 定义一个 nn.Sequential 的模型，包含一系列卷积操作。
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=False),
            # 卷积层：输入通道数为 ch_in，输出通道数为 ch_out，卷积核大小为 3x3，步幅为 1，边界填充为 1，不包含偏置。
            nn.BatchNorm2d(ch_out),  # 批归一化层：输入通道数为 ch_out。
            nn.ReLU(inplace=True),  # 激活函数：使用 ReLU。
            SplAtConv2d(ch_out, ch_out, kernel_size=3, padding=1, groups=2, radix=2, norm_layer=nn.BatchNorm2d),
            # 分组卷积：输入通道数为 ch_out，输出通道数为 ch_out，卷积核大小为 3x3，边界填充为 1，分为 2 组，使用 SplAtConv2d 实现。
            nn.ReLU(inplace=True),  # 再次使用 ReLU 激活。
        )
        self.downsample = nn.Sequential(  # 定义一个 nn.Sequential 的模型，包含一系列卷积操作。
            nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, bias=False),
            # 卷积层：输入通道数为 ch_in，输出通道数为 ch_out，卷积核大小为 1x1，步幅为 1，不包含偏置。
            nn.BatchNorm2d(ch_out),  # 批归一化层：输入通道数为 ch_out。
        )
        self.relu = nn.ReLU(inplace=True)  # 定义一个 ReLU 激活函数。
        # self.se = SELayer(ch_out)  # 使用 SELayer 实现 Squeeze-and-Excitation（SE）模块，提升模型的注意力机制。

    def forward(self, x):  # 定义前向传播函数，传入输入张量 x。
        residual = self.downsample(x)  # 使用 self.downsample 对输入 x 进行下采样，并将结果赋值为 residual，起到了残差的作用。
        out = self.conv(x)  # 对输入张量 x 进行一系列卷积操作（self.conv），并将结果赋值为 out。
        # out = SELayer(out)  # 使用 SELayer 实现 Squeeze-and-Excitation（SE）模块，提升模型的注意力机制。
        return self.relu(out + residual)  # 将 out 与 residual 相加，并通过 self.relu 进行激活，最终得到输出的张量。


class up_conv(nn.Module):  # 定义一个名为 up_conv 的类，该类继承自 nn.Module。
    def __init__(self, ch_in, ch_out):  # 类的初始化函数，传入两个参数：输入通道数 ch_in 和输出通道数 ch_out。
        super(up_conv, self).__init__()  # 调用父类（nn.Module）的 __init__() 方法。
        self.up = nn.Sequential(  # 定义一个 nn.Sequential 的模型，包含一系列卷积操作。
            nn.Upsample(scale_factor=2),    # 上采样：将输入张量的大小扩大两倍。
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),  # 卷积层：输入通道数为 ch_in，输出通道数为 ch_out，卷积核大小为 3x3，步幅为 1，边界填充为 1，包含偏置。
            nn.BatchNorm2d(ch_out),  # 批归一化层：输入通道数为 ch_out。
            nn.ReLU(inplace=True)   # 激活函数：使用 ReLU。
            # nn.ConvTranspose2d(ch_in, ch_out, kernel_size=2, stride=2)
            # 反卷积层：输入通道数为 ch_in，输出通道数为 ch_out，卷积核大小为 2x2，步幅为 2，用于上采样操作。
        )

    def forward(self, x):  # 定义前向传播函数，传入输入张量 x。
        x = self.up(x)  # 对输入张量 x 使用 self.up 进行上采样操作。
        return x  # 返回上采样后的张量。


# #########--------- Networks ---------#########
class SRF_UNet(nn.Module):  # 定义一个名为SRF_UNet的继承自nn.Module的类，用于构建神经网络模型
    def __init__(self, img_ch=3, output_ch=1):  # 初始化函数，接收输入图像通道数和输出通道数作为参数
        super(SRF_UNet, self).__init__()  # 调用父类的构造函数
        filters = [64, 128, 256, 512]  # 定义filters列表
        resnet = resnest.resnest50(pretrained=True)  # 使用resnest.resnest50()函数构造一个ResNeSt50模型
        self.firstconv = resnet.conv1  # 将模型的第一个卷积层赋值给self.firstconv
        self.firstbn = resnet.bn1  # 将模型的第一个BN层赋值给self.firstbn
        self.firstrelu = resnet.relu  # 将模型的第一个ReLU激活函数层赋值给self.firstrelu
        self.firstmaxpool = resnet.maxpool  # 将模型的第一个最大池化层赋值给self.firstmaxpool
        self.encoder1 = resnet.layer1  # 将模型的第一个ResNet stage赋值给self.encoder1
        self.encoder2 = resnet.layer2  # 将模型的第二个ResNet stage赋值给self.encoder2
        self.encoder3 = resnet.layer3  # 将模型的第三个ResNet stage赋值给self.encoder3
        self.encoder4 = resnet.layer4  # 将模型的第四个ResNet stage赋值给self.encoder4

        # 以下是上采样和下采样路径的构造
        self.Up5_thick = up_conv(ch_in=2048, ch_out=1024)  # 构造上采样路径5
        self.Up_conv5_thick = res_conv_block(ch_in=2048, ch_out=1024)  # 构造上采样路径5的卷积层
        self.cmba5=CBMAAttention(1024)
        self.Up4_thick = up_conv(ch_in=1024, ch_out=512)  # 构造上采样路径4
        self.Up_conv4_thick = res_conv_block(ch_in=1024, ch_out=512)  # 构造上采样路径4的卷积层
        self.cmba4=CBMAAttention(512)
        self.Up3_thick = up_conv(ch_in=512, ch_out=256)  # 构造上采样路径3
        self.Up_conv3_thick = res_conv_block(ch_in=512, ch_out=256)  # 构造上采样路径3的卷积层
        self.cmba3=CBMAAttention(256)
        self.Up2_thick = up_conv(ch_in=256, ch_out=64)  # 构造上采样路径2
        self.Up_conv2_thick = res_conv_block(ch_in=128, ch_out=64)  # 构造上采样路径2的卷积层
        self.cmba2=CBMAAttention(64)
        self.Up1_thick = up_conv(ch_in=64, ch_out=64)  # 构造上采样路径1
        self.Up_conv1_thick = res_conv_block(ch_in=64, ch_out=32)  # 构造上采样路径1的卷积层
        self.cmba1=CBMAAttention(32)
        self.Conv_1x1_thick = nn.Conv2d(32, output_ch, kernel_size=1)  # 定义一个1x1卷积层

    def forward(self, x):  # 前向传播函数，接收输入x作为参数
        x0 = self.firstconv(x)  # 获取第1个卷积层的输出
        x0 = self.firstbn(x0)  # 获取第1个BN层的输出
        x0 = self.firstrelu(x0)  # 获取第1个ReLU激活函数层的输出
        x1 = self.firstmaxpool(x0)  # 获取第1个最大池化层的输出

        x2 = self.encoder1(x1)  # 获取第1个ResNet stage的输出
        x3 = self.encoder2(x2)  # 获取第2个ResNet stage的输出
        x4 = self.encoder3(x3)  # 获取第3个ResNet stage的输出

        # 如果x4的高或宽不是偶数，则对其进行补0操作
        down_pad = False
        right_pad = False
        if x4.size()[2] % 2 == 1:
            x4 = F.pad(x4, (0, 0, 0, 1))
            down_pad = True
        if x4.size()[3] % 2 == 1:
            x4 = F.pad(x4, (0, 1, 0, 0))
            right_pad = True

        x5 = self.encoder4(x4)  # 获取第4个ResNet stage的输出

        # decoding + concat path
        d5_thick = self.Up5_thick(x5)  # 对x5进行上采样操作，获取上采样路径4的输出
        d5_thick = torch.cat((x4, d5_thick), dim=1)  # 对上采样路径4的输出和编码输出x4进行拼接操作

        # 如果x4经过补0操作，需要进行进一步的处理
        if down_pad and (not right_pad):
            d5_thick = d5_thick[:, :, :-1, :]
        if (not down_pad) and right_pad:
            d5_thick = d5_thick[:, :, :, :-1]
        if down_pad and right_pad:
            d5_thick = d5_thick[:, :, :-1, :-1]

        d5_thick = self.Up_conv5_thick(d5_thick)  # 对拼接输出进行卷积操作
        d5_thick=self.cmba5(d5_thick)

        d4_thick = self.Up4_thick(d5_thick)  # 对d5_thick进行上采样操作，获取上采样路径3的输出
        d4_thick = torch.cat((x3, d4_thick), dim=1)  # 对上采样路径3的输出和编码输出x3进行拼接操作
        d4_thick = self.Up_conv4_thick(d4_thick)  # 对拼接输出进行卷积操作
        d4_thick =self.cmba4(d4_thick)

        d3_thick = self.Up3_thick(d4_thick)  # 对d4_thick进行上采样操作，获取上采样路径2的输出
        d3_thick = torch.cat((x2, d3_thick), dim=1)  # 对上采样路径2的输出和编码输出x2进行拼接操作
        d3_thick = self.Up_conv3_thick(d3_thick)
        d3_thick=self.cmba3(d3_thick)

        d2_thick = self.Up2_thick(d3_thick)
        d2_thick = torch.cat((x0, d2_thick), dim=1)
        d2_thick = self.Up_conv2_thick(d2_thick)
        d2thick=self.cmba2(d2_thick)

        d1_thick = self.Up1_thick(d2_thick)
        # d1_thick = torch.cat((x, d1_thick), dim=1)
        d1_thick = self.Up_conv1_thick(d1_thick)
        d1_thick=self.cmba1(d1_thick)
        d1_thick = self.Conv_1x1_thick(d1_thick)

        out = nn.Sigmoid()(d1_thick)

        return out
        """
        d5_thin = self.Up5_thin(x5)
        d5_thin = torch.cat((x4, d5_thin), dim=1)

        d5_thin = self.Up_conv5_thin(d5_thin)

        d4_thin = self.Up4_thin(d5_thin)
        d4_thin = torch.cat((x3, d4_thin), dim=1)
        d4_thin = self.Up_conv4_thin(d4_thin)

        d3_thin = self.Up3_thin(d4_thin)  # x3
        d3_thin = torch.cat((x2, d3_thin), dim=1)
        d3_thin = self.Up_conv3_thin(d3_thin)
        """
