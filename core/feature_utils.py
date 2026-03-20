import json
from pathlib import Path

import librosa
import numpy as np

FEATURE_SPECS = {
    "mfcc": {"input_channels": 13},
    "mel": {"input_channels": 128},
    "mfcc_mel": {"input_channels": 141},
}

DEFAULT_INFERENCE_CONFIG = {
    "model_type": "single",
    "feature_type": "mfcc",
    "fusion_type": "concat",
    "target_sr": 22050,
    "n_mfcc": 13,
    "n_mels": 128,
    "max_length": 1000,
    "num_classes": 10,
}


def standardize_features(features):
    mean = np.mean(features)
    std = np.std(features)
    return (features - mean) / (std + 1e-8)


def pad_or_truncate_features(features, max_length=1000):
    if features.shape[1] < max_length:
        features = np.pad(features, ((0, 0), (0, max_length - features.shape[1])), mode="constant")
    else:
        features = features[:, :max_length]
    return features.astype(np.float32)


def get_input_channels(feature_type, n_mfcc=13, n_mels=128):
    if feature_type == "mfcc":
        return n_mfcc
    if feature_type == "mel":
        return n_mels
    if feature_type == "mfcc_mel":
        return n_mfcc + n_mels
    raise ValueError(f"Unsupported feature_type: {feature_type}")


def extract_audio_features(audio, sr, feature_type="mfcc", n_mfcc=13, n_mels=128, max_length=1000, standardize=False):
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
        if standardize:
            mfccs = standardize_features(mfccs)
            mel_db = standardize_features(mel_db)
        features = np.concatenate([mfccs, mel_db], axis=0)
    else:
        raise ValueError(
            f"Unsupported feature_type: {feature_type}. "
            f"Choose from: {', '.join(FEATURE_SPECS.keys())}"
        )

    if standardize and feature_type != "mfcc_mel":
        features = standardize_features(features)

    return pad_or_truncate_features(features, max_length=max_length)


def extract_dual_branch_features(audio, sr, n_mfcc=13, n_mels=128, max_length=1000, standardize=True):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)

    if standardize:
        mfccs = standardize_features(mfccs)
        mel_db = standardize_features(mel_db)

    mfccs = pad_or_truncate_features(mfccs, max_length=max_length)
    mel_db = pad_or_truncate_features(mel_db, max_length=max_length)
    return mfccs, mel_db


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