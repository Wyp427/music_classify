import os

import librosa
import numpy as np
from tqdm import tqdm

from feature_utils import extract_audio_features, extract_dual_branch_features

#数据预处理
def audio_to_cnn_data(
    folder_path,
    target_sr=22050,
    n_mfcc=13,
    n_mels=128,
    max_length=1000,
    feature_type="mfcc",
    model_type="single",
    standardize=False,
    cache_file=None,
):
    if cache_file and os.path.exists(cache_file):
        print(f"加载缓存数据: {cache_file}...")
        cache = np.load(cache_file)
        if model_type == "dual_branch":
            return cache["mfcc_data"], cache["mel_data"], cache["encoded_labels"]
        return cache["data"], cache["encoded_labels"]

    audio_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.au') or file.endswith('.wav') or file.endswith('.mp3'):
                audio_files.append(os.path.join(root, file))

    labels = []
    if model_type == "dual_branch":
        mfcc_data = []
        mel_data = []
    else:
        data = []

    for file_path in tqdm(sorted(audio_files), desc="Processing audio files"):
        try:
            audio, sr = librosa.load(file_path, sr=target_sr)
            if model_type == "dual_branch":
                mfcc_features, mel_features = extract_dual_branch_features(
                    audio,
                    sr,
                    n_mfcc=n_mfcc,
                    n_mels=n_mels,
                    max_length=max_length,
                    standardize=standardize,
                )
                mfcc_data.append(mfcc_features)
                mel_data.append(mel_features)
            else:
                features = extract_audio_features(
                    audio,
                    sr,
                    feature_type=feature_type,
                    n_mfcc=n_mfcc,
                    n_mels=n_mels,
                    max_length=max_length,
                    standardize=standardize,
                )
                data.append(features)

            label = os.path.basename(os.path.dirname(file_path))
            labels.append(label)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    unique_labels = np.unique(labels)
    label_dict = {label: i for i, label in enumerate(unique_labels)}
    encoded_labels = np.array([label_dict[label] for label in labels])

    if model_type == "dual_branch":
        mfcc_data = np.expand_dims(np.array(mfcc_data, dtype=np.float32), axis=-1)
        mel_data = np.expand_dims(np.array(mel_data, dtype=np.float32), axis=-1)
        if cache_file:
            print(f"保存缓存数据到: {cache_file}...")
            np.savez(cache_file, mfcc_data=mfcc_data, mel_data=mel_data, encoded_labels=encoded_labels)
        return mfcc_data, mel_data, encoded_labels

    data = np.expand_dims(np.array(data, dtype=np.float32), axis=-1)
    if cache_file:
        print(f"保存缓存数据到: {cache_file}...")
        np.savez(cache_file, data=data, encoded_labels=encoded_labels)
    return data, encoded_labels