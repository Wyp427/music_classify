import librosa
import torch
import numpy as np

from feature_utils import extract_audio_features

"""
pre_process.py

该文件用于音频分类模型的推理前处理，主要负责对单个音频输入进行特征提取、
数据转换及模型预测，是模型推理流程中的关键模块。

本模块通过参数化设计，实现对多种音频特征（MFCC / Mel / MFCC+Mel）的统一支持，
并确保推理阶段与训练阶段的特征配置一致。

主要功能如下：

1. 统一推理逻辑封装（_predict_from_audio）
------------------------------------------------
新增内部函数 _predict_from_audio(...)，用于抽取单音频预测的公共流程，包括：
- 音频特征提取（调用统一特征提取函数）
- 特征转为 Tensor 并调整维度
- 自动迁移到模型所在设备（CPU / GPU）
- 前向传播（forward）
- Softmax 计算类别概率

通过该封装，避免在多个推理接口中重复实现相同逻辑，提高代码复用性与可维护性。


2. 推理接口参数化（支持多特征类型）
------------------------------------------------
推理函数（文件路径版本 / 文件对象版本）均支持以下参数：
- feature_type：特征类型（mfcc / mel / mfcc_mel）
- n_mfcc：MFCC维度
- n_mels：Mel滤波器数量
- max_length：时间序列长度

该设计确保推理阶段使用与训练阶段一致的特征提取方式，
避免因特征维度不一致导致模型预测错误。


3. 自动设备适配（CPU / GPU）
------------------------------------------------
在推理过程中，通过以下方式获取模型所在设备：

    device = next(model.parameters()).device

并将输入特征张量自动迁移至该设备：

    features = features.to(device)

从而避免因输入与模型不在同一设备上导致的运行错误（device mismatch），
提升系统的稳定性与通用性。


总结：
------------------------------------------------
本模块通过对推理流程的统一封装，实现了：
- 单音频预测流程的模块化与复用
- 多种音频特征类型的灵活支持
- 推理与训练阶段特征配置的一致性保障
- 自动适配CPU/GPU环境，提高系统鲁棒性

为后续系统部署（Web服务 / API接口）及多模态扩展提供了稳定的推理基础。
"""



def _predict_from_audio(model, audio, sr, feature_type="mfcc", n_mfcc=13, n_mels=128, max_length=1000):
    features = extract_audio_features(
        audio,
        sr,
        feature_type=feature_type,
        n_mfcc=n_mfcc,
        n_mels=n_mels,
        max_length=max_length,
    )

    features = np.expand_dims(features, axis=-1)
    features = torch.tensor(features, dtype=torch.float32)
    features = features.unsqueeze(0)

    device = next(model.parameters()).device
    features = features.to(device)

    with torch.no_grad():
        prediction = model(features)

    predicted_class = torch.argmax(prediction, dim=1).item()
    probabilities = torch.nn.functional.softmax(prediction, dim=1).squeeze(0).cpu().numpy()
    return predicted_class, probabilities


def preprocess_and_predict_file(
    model,
    music_file,
    target_sr=22050,
    n_mfcc=13,
    n_mels=128,
    max_length=1000,
    feature_type="mfcc",
):
    try:
        music_file.seek(0)
        audio, sr = librosa.load(music_file, sr=target_sr)
        return _predict_from_audio(
            model,
            audio,
            sr,
            feature_type=feature_type,
            n_mfcc=n_mfcc,
            n_mels=n_mels,
            max_length=max_length,
        )
    except Exception as e:
        print(f"Error processing the audio file: {e}")
        return None, None


def preprocess_and_predict(
    model,
    file_path,
    target_sr=22050,
    n_mfcc=13,
    n_mels=128,
    max_length=1000,
    feature_type="mfcc",
):
    try:
        audio, sr = librosa.load(file_path, sr=target_sr)
        return _predict_from_audio(
            model,
            audio,
            sr,
            feature_type=feature_type,
            n_mfcc=n_mfcc,
            n_mels=n_mels,
            max_length=max_length,
        )
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None