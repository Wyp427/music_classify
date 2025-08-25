import os
import librosa
import numpy as np
from tqdm import tqdm

def audio_to_cnn_data(folder_path, target_sr=22050, n_mfcc=13, max_length=1000, cache_file=None):
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
            if file.endswith('.au'):
                au_files.append(os.path.join(root, file))

    data = []
    labels = []
    for file_path in tqdm(au_files, desc="Processing audio files"):
        try:
            audio, sr = librosa.load(file_path, sr=target_sr)
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
            if mfccs.shape[1] < max_length:
                mfccs = np.pad(mfccs, ((0, 0), (0, max_length - mfccs.shape[1])), mode='constant')
            else:
                mfccs = mfccs[:, :max_length]
            data.append(mfccs)
            label = os.path.basename(os.path.dirname(file_path))
            labels.append(label)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    data = np.array(data)
    data = np.expand_dims(data, axis=-1)
    unique_labels = np.unique(labels)
    label_dict = {label: i for i, label in enumerate(unique_labels)}
    encoded_labels = np.array([label_dict[label] for label in labels])

    # 如果提供了缓存文件路径，将数据保存为缓存文件
    if cache_file:
        print(f"保存缓存数据到: {cache_file}...")
        np.savez(cache_file, data=data, encoded_labels=encoded_labels)

    return data, encoded_labels
