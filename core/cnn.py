import torch
import torch.nn as nn
import torch.nn.functional as F

"""
本文件定义音频分类使用的卷积神经网络模型（AudioCNN）。 文件说明：

主要改动是引入 input_channels 参数，使模型能够动态适配不同类型的音频特征输入，
不再固定为传统 MFCC 的 13 维输入，从而支持 Mel（128维）以及 MFCC+Mel（141维）等多种特征形式。

第一层卷积根据 input_channels 自动构建，保证输入特征维度与网络结构一致，
避免特征切换时出现维度不匹配问题。

除输入层外，模型保留原有结构设计，包括卷积模块、Mish 激活函数、
SE 注意力机制、残差结构、全局平均池化、Dropout 以及全连接分类层。

该设计保证在不同特征实验中仅改变输入形式，而保持网络结构一致，
从而使实验对比更加公平和可控。
"""


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
        out = Mish()(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        out = Mish()(out)
        return out


class AudioCNN(nn.Module):
    def __init__(self, num_classes=10, input_channels=13):
        super(AudioCNN, self).__init__()
        self.in_channels = 64
        self.input_channels = input_channels
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 4, stride=1, groups=1)
        self.layer2 = self._make_layer(128, 4, stride=2, groups=2)
        self.layer3 = self._make_layer(256, 4, stride=2, groups=4)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(256, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.3)
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
        out = Mish()(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.global_avg_pool(out)
        out = torch.flatten(out, 1)
        out = Mish()(self.bn_fc1(self.fc1(out)))
        out = self.dropout1(out)
        out = Mish()(self.bn_fc2(self.fc2(out)))
        out = self.dropout2(out)
        out = self.fc3(out)
        return out