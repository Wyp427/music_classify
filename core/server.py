import io
import json
import tempfile

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import streamlit as st
import torch

from cnn import AudioCNN
from feature_utils import load_feature_config
from label_mapper import GTZANLabelMapper
from pre_process import preprocess_and_predict


"""
本文件实现基于 Streamlit 的音频分类可视化界面。

系统启动时自动加载 best_model_config.json，
并根据配置构建模型与推理环境，保证与训练阶段一致。

在界面侧边栏显示当前模型使用的特征类型，
便于用户在实验过程中区分不同特征模型。

用户上传音频后，调用统一推理接口 preprocess_and_predict，
并使用配置中的特征参数进行特征提取与预测。

预测结果中同时展示分类结果与当前特征类型，
方便进行实验记录与结果对比。

该模块实现了模型推理的可视化交互，支持快速验证不同特征方案效果。
"""



# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = load_feature_config("best_model_config.json")

# 加载模型
model = AudioCNN(
    num_classes=config["num_classes"],
    input_channels=config["input_channels"],
)
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.to(device)
model.eval()

with st.sidebar:
    st.title("菜单")
    st.write(f"当前特征类型: {config['feature_type']}")
    uploaded_file = st.file_uploader("上传音乐文件", type=["mp3", "wav", "ogg", "flac", "au"])

st.markdown("<h2 style='text-align: center;'>音乐流派分类 BY CNN:ACC >=87.5%</h2>", unsafe_allow_html=True)

json_data = None
audio_data = None
if uploaded_file is not None:
    if uploaded_file.name.endswith('.json'):
        try:
            json_data = json.load(uploaded_file)
        except json.JSONDecodeError:
            st.error("错误: 无法解析上传的 JSON 文件内容。")
    else:
        audio_data = uploaded_file

if json_data is None:
    try:
        with open('training_output.json', 'r', encoding='utf-8') as file:
            json_data = json.load(file)
    except FileNotFoundError:
        st.error("错误: 未找到 'training_output.json' 文件。")
    except json.JSONDecodeError:
        st.error("错误: 无法解析 'training_output.json' 文件的 JSON 内容。")


def convert_au_to_wav_librosa(audio_data):
    audio, sr = librosa.load(io.BytesIO(audio_data), sr=None)
    wav_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(wav_file.name, audio, sr)
    return wav_file.name


def display_genre_probabilities(probabilities):
    categories = ['布鲁斯', '古典', '乡村', '迪斯科', '嘻哈', '爵士', '金属', '流行', '雷鬼', '摇滚']
    col1, col2 = st.columns(2)
    with col1:
        for i in range(5):
            st.write(f"{categories[i]} - {probabilities[i] * 100:.2f}%")
    with col2:
        for i in range(5, 10):
            st.write(f"{categories[i]} - {probabilities[i] * 100:.2f}%")


if json_data is not None:
    epochs = [d["epoch"] for d in json_data]
    train_loss = [d["train_loss"] for d in json_data]
    train_accuracy = [d["train_accuracy"] for d in json_data]
    val_loss = [d["val_loss"] for d in json_data]
    val_accuracy = [d["val_accuracy"] for d in json_data]

    def plot_graph(epochs, data, title, x_label, y_label, color):
        fig = plt.figure(figsize=(2, 2))
        plt.plot(epochs, data, color)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        return fig

    train_loss_fig = plot_graph(epochs, train_loss, 'Train Loss', 'Epoch', 'Loss', 'r-')
    train_accuracy_fig = plot_graph(epochs, train_accuracy, 'Train Accuracy', 'Epoch', 'Accuracy (%)', 'g-')
    val_loss_fig = plot_graph(epochs, val_loss, 'Validation Loss', 'Epoch', 'Loss', 'b-')
    val_accuracy_fig = plot_graph(epochs, val_accuracy, 'Validation Accuracy', 'Epoch', 'Accuracy (%)', 'y-')

    col1, col2 = st.columns([3, 2])

    with col1:
        if audio_data is not None:
            try:
                audio_bytes = audio_data.read()
                wav_file_path = convert_au_to_wav_librosa(audio_bytes)

                audio, sr = librosa.load(wav_file_path)
                mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

                fig, ax = plt.subplots(figsize=(6, 3))
                librosa.display.specshow(mel_spec_db, x_axis='time', y_axis='mel', sr=sr, ax=ax)
                ax.set_title('Mel Spectrogram')
                ax.set_aspect('auto', adjustable='box')
                st.pyplot(fig)

                with open(wav_file_path, 'rb') as f:
                    st.audio(f.read(), format="audio/wav")
            except Exception as e:
                st.error(f"播放音频或生成 Mel 频谱图时出现错误: {e}")
        else:
            st.error("请上传一个音频文件以查看 Mel 频谱图。")

        if audio_data is not None:
            try:
                label_mapper = GTZANLabelMapper()
                predicted_class, probabilities = preprocess_and_predict(
                    model,
                    wav_file_path,
                    target_sr=config["target_sr"],
                    n_mfcc=config["n_mfcc"],
                    n_mels=config["n_mels"],
                    max_length=config["max_length"],
                    feature_type=config["feature_type"],
                )

                if predicted_class is not None and probabilities is not None:
                    predicted_label = label_mapper.get_label(predicted_class)
                    st.success(f"🎵 当前特征：**{config['feature_type']}**，预测音乐风格：**{predicted_label}**")
                    display_genre_probabilities(probabilities)
                else:
                    st.error("预测时返回了无效结果。")

            except Exception as e:
                st.error(f"预测时出现错误: {e}")

    with col2:
        sub_col1, sub_col2 = st.columns(2)
        with sub_col1:
            st.pyplot(train_loss_fig)
        with sub_col2:
            st.pyplot(train_accuracy_fig)

        sub_col3, sub_col4 = st.columns(2)
        with sub_col3:
            st.pyplot(val_loss_fig)
        with sub_col4:
            st.pyplot(val_accuracy_fig)