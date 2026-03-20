import io
import json
import tempfile

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import streamlit as st

from label_mapper import GTZANLabelMapper
from model_factory import load_model_and_config
from pre_process import preprocess_and_predict

#可视化代码文件

model, config, _ = load_model_and_config("best_model_config.json", "best_model.pth")

with st.sidebar:
    st.title("菜单")
    st.write(f"当前模型类型: {config.get('model_type', 'single')}")
    st.write(f"当前特征类型: {config.get('feature_type', 'mfcc')}")
    if config.get('model_type') == 'dual_branch':
        st.write(f"融合方式: {config.get('fusion_type', 'concat')}")
    uploaded_file = st.file_uploader("上传音乐文件", type=["mp3", "wav", "ogg", "flac", "au"])

st.markdown("<h2 style='text-align: center;'>音乐流派分类 BY CNN</h2>", unsafe_allow_html=True)

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


def convert_audio_to_wav(audio_bytes):
    audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)
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
                wav_file_path = convert_audio_to_wav(audio_bytes)
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
                    feature_type=config.get("feature_type", "mfcc"),
                    model_type=config.get("model_type", "single"),
                    standardize=config.get("standardize", False),
                )
                if predicted_class is not None and probabilities is not None:
                    predicted_label = label_mapper.get_label(predicted_class)
                    st.success(f"🎵 模型：**{config.get('model_type', 'single')}**，预测音乐风格：**{predicted_label}**")
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