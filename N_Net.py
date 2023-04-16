import torch
import torch.nn as nn
import torch.nn.functional as F
from splat import SplAtConv2d

class DoubleConv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_channels, out_channels):
        # 定义双层卷积模块
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            # 进行卷积运算
            nn.BatchNorm2d(out_channels),
            # 进行批标准化
            nn.ReLU(inplace=True),
            # 进行ReLU激活
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 模型的前向传播
        x = self.conv(x)
        return x
class up_conv(nn.Module):  # 定义一个名为 up_conv 的类，该类继承自 nn.Module。
    def __init__(self, ch_in, ch_out):  # 类的初始化函数，传入两个参数：输入通道数 ch_in 和输出通道数 ch_out。
        super(up_conv, self).__init__()  # 调用父类（nn.Module）的 __init__() 方法。
        self.up = nn.Sequential(  # 定义一个 nn.Sequential 的模型，包含一系列卷积操作。
            # nn.Upsample(scale_factor=2),    # 上采样：将输入张量的大小扩大两倍。
            # nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),  # 卷积层：输入通道数为 ch_in，输出通道数为 ch_out，卷积核大小为 3x3，步幅为 1，边界填充为 1，包含偏置。
            # nn.BatchNorm2d(ch_out),  # 批归一化层：输入通道数为 ch_out。
            # nn.ReLU(inplace=True)   # 激活函数：使用 ReLU。
            nn.ConvTranspose2d(ch_in, ch_out, kernel_size=2, stride=2)
            # 反卷积层：输入通道数为 ch_in，输出通道数为 ch_out，卷积核大小为 2x2，步幅为 2，用于上采样操作。
        )

    def forward(self, x):  # 定义前向传播函数，传入输入张量 x。
        x = self.up(x)  # 对输入张量 x 使用 self.up 进行上采样操作。
        return x  # 返回上采样后的张量。
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

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        # 反卷积或bilinear上采样+卷积连接
        super().__init__()

        # 判断是否使用bilinear上采样
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # 上采样方法
            self.conv = DoubleConv(in_channels, out_channels)
            # 双层卷积模块
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, 2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # 上采样得到的特征图x1和输入特征图x2在通道维度上进行拼接
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class NNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        # Unet模型定义，包含encoder和decoder
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 2048)
        # 编码层–输入大小(n_channels)到特征图大小(64)的第一层

        self.down1 = DoubleConv(2048, 2568)
        self.down2 = DoubleConv(2568, 3080)
        self.down3 = DoubleConv(3080, 3592)
        # self.down4 = DoubleConv(20384, 40786)
        # 编码层–特征图大小(64)到1024的一层，由四个反卷积模块实现
        self.up1 = Up(3592, 3080, bilinear)
        self.up2 = Up(3080, 2568, bilinear)
        self.up3 = Up(2568, 2048, bilinear)
        # self.up4 = Up(2048, 2048, bilinear)
        # 解码层（上采样层），由四个反卷积模块实现
        self.outc = nn.Conv2d(2048, n_classes, 1)
        # 输出层–将最后一层的特征图转换为n_classes种类的像素预测结果

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 模型的前向传播
        x1 = self.inc(x)
        x2 = self.down1(F.max_pool2d(x1, 2))
        x3 = self.down2(F.max_pool2d(x2, 2))
        x4 = self.down3(F.max_pool2d(x3, 2))
        x5 = self.down4(F.max_pool2d(x4, 2))
        print(x5.size())
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        out = nn.Sigmoid()(out)
        return out