import json
from pathlib import Path

import librosa
import numpy as np

"""
feature_utils.py

该文件用于统一管理音频特征提取及相关配置，是音频分类模型的数据处理核心模块。
通过对特征类型、输入维度及提取流程的封装，实现训练与推理阶段的一致性与可扩展性。

主要功能如下：

1. 特征类型与输入维度定义（FEATURE_SPECS）
------------------------------------------------
定义当前系统支持的音频特征类型及其对应的输入通道数，包括：
- mfcc：13维 MFCC 特征
- mel：128维 Mel 频谱特征
- mfcc_mel：MFCC 与 Mel 融合特征（共 141 通道）

该配置用于自动匹配 CNN 模型第一层的输入通道数，避免手动修改模型结构。


2. 推理默认配置（DEFAULT_INFERENCE_CONFIG）
------------------------------------------------
定义推理阶段的默认特征参数，包括：
- 特征类型（feature_type）
- 采样率（target_sr）
- MFCC维度（n_mfcc）
- Mel滤波器数量（n_mels）
- 时间长度（max_length）
- 分类类别数（num_classes）

当未加载配置文件时，系统使用该默认配置，保证推理过程稳定运行。


3. 输入通道获取函数（get_input_channels）
------------------------------------------------
根据当前选用的特征类型（mfcc / mel / mfcc_mel），
自动返回模型所需的输入通道数，用于构建CNN模型。


4. 音频特征提取函数（extract_audio_features）
------------------------------------------------
统一实现音频特征提取流程，支持三种模式：

- mfcc：
  使用 librosa 提取 MFCC 特征

- mel：
  提取 Mel 频谱，并转换为对数功率谱（dB）

- mfcc_mel：
  同时提取 MFCC 与 Mel 特征，并沿通道维进行拼接

此外，对时间维进行统一处理：
- 长度不足：进行零填充（padding）
- 长度过长：进行裁剪（truncation）

确保所有输入样本尺寸一致，满足 CNN 输入要求。


5. 特征配置持久化（save_feature_config / load_feature_config）
------------------------------------------------
用于保存与加载特征配置文件（JSON格式），记录当前模型训练所使用的特征参数。

其作用是：
- 保证训练与推理阶段使用一致的特征设置
- 避免因特征维度不匹配导致模型加载或预测失败
- 提高系统的可维护性与可复现性


总结：
------------------------------------------------
本模块通过对音频特征提取与配置管理的统一封装，实现了：
- 多特征类型（MFCC / Mel / 融合）的灵活切换
- 模型输入维度的自动适配
- 训练与推理流程的一致性保障

为后续多模态融合（音频 + 歌词）及系统扩展提供了良好的基础。
"""

FEATURE_SPECS = {
    "mfcc": {"input_channels": 13},
    "mel": {"input_channels": 128},
    "mfcc_mel": {"input_channels": 141},
}

DEFAULT_INFERENCE_CONFIG = {
    "feature_type": "mfcc",
    "target_sr": 22050,
    "n_mfcc": 13,
    "n_mels": 128,
    "max_length": 1000,
    "num_classes": 10,
}


def get_input_channels(feature_type, n_mfcc=13, n_mels=128):
    if feature_type == "mfcc":
        return n_mfcc
    if feature_type == "mel":
        return n_mels
    if feature_type == "mfcc_mel":
        return n_mfcc + n_mels
    raise ValueError(f"Unsupported feature_type: {feature_type}")


def extract_audio_features(audio, sr, feature_type="mfcc", n_mfcc=13, n_mels=128, max_length=1000):
    """提取可配置音频特征，并统一裁剪/补齐到固定长度。"""
    feature_type = feature_type.lower()

    if feature_type == "mfcc":
        features = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    elif feature_type == "mel":
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
        features = librosa.power_to_db(mel_spec, ref=np.max)
    elif feature_type == "mfcc_mel":
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        features = np.concatenate([mfccs, mel_db], axis=0)
    else:
        raise ValueError(
            f"Unsupported feature_type: {feature_type}. "
            f"Choose from: {', '.join(FEATURE_SPECS.keys())}"
        )

    if features.shape[1] < max_length:
        features = np.pad(features, ((0, 0), (0, max_length - features.shape[1])), mode="constant")
    else:
        features = features[:, :max_length]

    return features.astype(np.float32)


def save_feature_config(config_path, config):
    config_path = Path(config_path)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)


def load_feature_config(config_path, default=None):
    config_path = Path(config_path)
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return (default or DEFAULT_INFERENCE_CONFIG).copy()