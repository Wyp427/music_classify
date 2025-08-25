import librosa
import torch
import numpy as np

def preprocess_and_predict_file(model, music_file, target_sr=22050, n_mfcc=13, max_length=1000):
    try:
        # 从音频文件对象读取音频数据
        music_file.seek(0)  # 确保文件指针在开始位置
        audio, sr = librosa.load(music_file, sr=target_sr)

        # 提取 MFCC 特征
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

        # 对 MFCC 特征进行填充或裁剪
        if mfccs.shape[1] < max_length:
            mfccs = np.pad(mfccs, ((0, 0), (0, max_length - mfccs.shape[1])), mode='constant')
        else:
            mfccs = mfccs[:, :max_length]

        # 转换为 PyTorch 张量并调整形状
        mfccs = np.expand_dims(mfccs, axis=-1)  # 添加通道维度
        mfccs = torch.tensor(mfccs, dtype=torch.float32)  # 转换为张量
        mfccs = mfccs.unsqueeze(0)  # 添加批次维度 (batch_size, n_mfcc, max_length, 1)

        # 使用模型进行预测
        with torch.no_grad():  # 在推理过程中禁用梯度计算
            prediction = model(mfccs)  # 直接调用模型进行前向传播

        # 获取预测的类别索引
        predicted_class = torch.argmax(prediction, dim=1).item()

        # 获取所有类别的概率分布
        probabilities = torch.nn.functional.softmax(prediction, dim=1).squeeze(0).cpu().numpy()

        return predicted_class, probabilities

    except Exception as e:
        print(f"Error processing the audio file: {e}")
        return None, None

def preprocess_and_predict(model, file_path, target_sr=22050, n_mfcc=13, max_length=1000):
    try:
        # 加载音频文件
        audio, sr = librosa.load(file_path, sr=target_sr)

        # 提取 MFCC 特征
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

        # 对 MFCC 特征进行填充或裁剪
        if mfccs.shape[1] < max_length:
            mfccs = np.pad(mfccs, ((0, 0), (0, max_length - mfccs.shape[1])), mode='constant')
        else:
            mfccs = mfccs[:, :max_length]

        # 转换为 PyTorch 张量并调整形状
        mfccs = np.expand_dims(mfccs, axis=-1)  # 添加通道维度
        mfccs = torch.tensor(mfccs, dtype=torch.float32)  # 转换为张量
        mfccs = mfccs.unsqueeze(0)  # 添加批次维度 (batch_size, n_mfcc, max_length, 1)

        # 使用模型进行预测
        with torch.no_grad():  # 在推理过程中禁用梯度计算
            prediction = model(mfccs)  # 直接调用模型进行前向传播

        # 获取预测的类别索引
        predicted_class = torch.argmax(prediction, dim=1).item()

        # 获取所有类别的概率分布
        probabilities = torch.nn.functional.softmax(prediction, dim=1).squeeze(0).cpu().numpy()

        return predicted_class, probabilities

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None
