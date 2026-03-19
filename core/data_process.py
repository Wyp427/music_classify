import os

import librosa
import numpy as np
from tqdm import tqdm

from feature_utils import extract_audio_features

"""
本文件用于训练前的数据预处理与特征构建，将原始音频数据转换为 CNN 可用的输入格式。

在原有基础上引入 feature_type、n_mfcc、n_mels、max_length 等参数，
支持根据配置灵活选择 MFCC、Mel 或组合特征进行批量处理。

特征提取统一调用 feature_utils 中的 extract_audio_features 函数，
保证训练阶段与推理阶段使用完全一致的特征构造逻辑，避免不一致问题。

同时扩展支持多种音频格式（.au、.wav、.mp3），增强数据兼容性。

对输入音频文件路径进行排序处理，以保证数据处理顺序稳定，
提高实验结果的可复现性。

此外支持缓存机制，根据特征类型生成独立缓存文件，
避免不同特征实验之间的数据混用，提高实验效率。
"""


def audio_to_cnn_data(
    folder_path,
    target_sr=22050,
    n_mfcc=13,
    n_mels=128,
    max_length=1000,
    feature_type="mfcc",
    cache_file=None,
):
    # 如果缓存文件存在，则直接加载缓存
    if cache_file and os.path.exists(cache_file):
        print(f"加载缓存数据: {cache_file}...")
        cache = np.load(cache_file)
        data = cache["data"]
        encoded_labels = cache["encoded_labels"]
        return data, encoded_labels

    au_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.au') or file.endswith('.wav') or file.endswith('.mp3'):
                au_files.append(os.path.join(root, file))

    data = []
    labels = []
    for file_path in tqdm(sorted(au_files), desc="Processing audio files"):
        try:
            audio, sr = librosa.load(file_path, sr=target_sr)
            features = extract_audio_features(
                audio,
                sr,
                feature_type=feature_type,
                n_mfcc=n_mfcc,
                n_mels=n_mels,
                max_length=max_length,
            )
            data.append(features)
            label = os.path.basename(os.path.dirname(file_path))
            labels.append(label)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    data = np.array(data, dtype=np.float32)
    data = np.expand_dims(data, axis=-1)
    unique_labels = np.unique(labels)
    label_dict = {label: i for i, label in enumerate(unique_labels)}
    encoded_labels = np.array([label_dict[label] for label in labels])

    # 如果提供了缓存文件路径，将数据保存为缓存文件
    if cache_file:
        print(f"保存缓存数据到: {cache_file}...")
        np.savez(cache_file, data=data, encoded_labels=encoded_labels)

    return data, encoded_labels