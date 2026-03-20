import torch
import torch.nn as nn

from cnn import ResidualBlock, Mish

"""
dual_branch_cnn.py  文件说明

该文件定义了标准化双分支音频特征融合模型，用于同时建模 MFCC 特征和 Mel 频谱特征。
模型由两个并行的卷积分支组成，分别提取两类音频特征的高层表示，
再通过特征拼接（concat）或门控融合（gated）的方式进行特征融合，
最后通过全连接分类层输出音乐风格类别结果。

其中：
1. AudioBranchCNN 用于对单一音频特征分支进行卷积特征提取；
2. DualBranchFusionCNN 用于完成双分支特征融合与最终分类。

该文件主要服务于“标准化 + 双分支 MFCC + Mel 融合模型”的实验设计，
用于验证双分支结构是否优于简单特征拼接方式。
"""



class AudioBranchCNN(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 2, stride=1, groups=1)
        self.layer2 = self._make_layer(128, 2, stride=2, groups=2)
        self.layer3 = self._make_layer(256, 2, stride=2, groups=4)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 128)
        self.bn_fc = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.3)

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
        out = Mish()(self.bn_fc(self.fc(out)))
        out = self.dropout(out)
        return out


class DualBranchFusionCNN(nn.Module):
    def __init__(self, num_classes=10, mfcc_channels=13, mel_channels=128, fusion_type="concat"):
        super().__init__()
        self.fusion_type = fusion_type
        self.mfcc_branch = AudioBranchCNN(mfcc_channels)
        self.mel_branch = AudioBranchCNN(mel_channels)

        if fusion_type == "concat":
            fusion_dim = 256
        elif fusion_type == "gated":
            self.gate = nn.Sequential(
                nn.Linear(256, 128),
                nn.Sigmoid(),
            )
            fusion_dim = 128
        else:
            raise ValueError(f"Unsupported fusion_type: {fusion_type}")

        self.fusion_fc1 = nn.Linear(fusion_dim, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.3)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, mfcc_x, mel_x):
        h_mfcc = self.mfcc_branch(mfcc_x)
        h_mel = self.mel_branch(mel_x)

        if self.fusion_type == "concat":
            fused = torch.cat([h_mfcc, h_mel], dim=1)
        else:
            gate_input = torch.cat([h_mfcc, h_mel], dim=1)
            gate = self.gate(gate_input)
            fused = gate * h_mfcc + (1 - gate) * h_mel

        fused = Mish()(self.bn_fc1(self.fusion_fc1(fused)))
        fused = self.dropout1(fused)
        return self.classifier(fused)