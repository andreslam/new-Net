import torch.nn as nn

class CBMAAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(CBMAAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(1)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        out = out * x

        spatial_out = self.conv(out)
        spatial_out = self.bn(spatial_out)
        spatial_out = self.relu2(spatial_out)

        out = out * spatial_out.expand_as(out)
        return self.sigmoid(out)映射到0到1之间。最后，将输入张量与映射后的特征向量相乘，得到输出张量。
