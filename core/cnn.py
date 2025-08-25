import torch
import torch.nn as nn
import torch.nn.functional as F


# Mish 激活函数定义
class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


# SE 模块
class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            Mish(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# 修改后的 ResidualBlock
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False, groups=groups)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False, groups=groups)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEModule(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = Mish()(self.bn1(self.conv1(x)))  # 使用 Mish 激活函数
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        out = Mish()(out)  # 使用 Mish 激活函数
        return out


# 修改后的 AudioCNN 模型
class AudioCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(AudioCNN, self).__init__()
        self.in_channels = 64  # 初始通道数
        self.conv1 = nn.Conv2d(13, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 4, stride=1, groups=1)  # 增加残差块数量
        self.layer2 = self._make_layer(128, 4, stride=2, groups=2)  # 增加残差块数量，使用分组卷积
        self.layer3 = self._make_layer(256, 4, stride=2, groups=4)  # 增加残差块数量，使用分组卷积
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化
        self.fc1 = nn.Linear(256, 256)  # 增加全连接层神经元数量
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.3)  # Dropout 概率
        self.fc2 = nn.Linear(256, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride, groups):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, stride, groups))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = Mish()(self.bn1(self.conv1(x)))  # 使用 Mish 激活函数
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.global_avg_pool(out)
        out = torch.flatten(out, 1)
        out = Mish()(self.bn_fc1(self.fc1(out)))  # 使用 Mish 激活函数
        out = self.dropout1(out)
        out = Mish()(self.bn_fc2(self.fc2(out)))
        out = self.dropout2(out)
        out = self.fc3(out)
        return out
