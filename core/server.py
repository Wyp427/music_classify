import io
import json
import tempfile
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import streamlit as st

from model_factory import load_model_and_config
from pre_process import predict_lyrics, preprocess_and_predict

AUDIO_CONFIG_PATH = Path("best_model_config.json")
AUDIO_MODEL_PATH = Path("best_model.pth")
LYRICS_CONFIG_PATH = Path("lyrics_best_model_config.json")
LYRICS_MODEL_PATH = Path("lyrics_best_model.pth")


def load_bundle(config_path, model_path):
    if config_path.exists() and model_path.exists():
        return load_model_and_config(str(config_path), str(model_path))
    return None, {}, None, None


audio_model, audio_config, _, audio_label_mapper = load_bundle(AUDIO_CONFIG_PATH, AUDIO_MODEL_PATH)
lyrics_model, lyrics_config, _, lyrics_label_mapper = load_bundle(LYRICS_CONFIG_PATH, LYRICS_MODEL_PATH)


def load_training_json(config, fallback_name):
    training_path = Path(config.get("training_output_path", fallback_name))
    if training_path.exists():
        try:
            return json.loads(training_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            st.error(f"错误: 无法解析 {training_path.name} 文件内容。")
    return []


def convert_audio_to_wav(audio_bytes):
    audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)
    wav_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(wav_file.name, audio, sr)
    return wav_file.name


def display_genre_probabilities(probabilities, label_mapper):
    labels = label_mapper.get_labels() if label_mapper is not None else [str(i) for i in range(len(probabilities))]
    midpoint = (len(labels) + 1) // 2
    col1, col2 = st.columns(2)
    with col1:
        for i in range(midpoint):
            st.write(f"{labels[i]} - {probabilities[i] * 100:.2f}%")
    with col2:
        for i in range(midpoint, len(labels)):
            st.write(f"{labels[i]} - {probabilities[i] * 100:.2f}%")


def plot_graph(epochs, data, title, x_label, y_label, color):
    fig = plt.figure(figsize=(3, 2.4))
    plt.plot(epochs, data, color)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    return fig


st.set_page_config(page_title="音乐流派分类系统", layout="wide")

with st.sidebar:
    st.title("菜单")
    st.subheader("音频模型")
    if audio_config:
        st.write(f"当前模型类型: {audio_config.get('model_type', 'single')}")
        st.write(f"当前特征类型: {audio_config.get('feature_type', 'mfcc')}")
        if audio_config.get('model_type') == 'dual_branch':
            st.write(f"融合方式: {audio_config.get('fusion_type', 'concat')}")
    else:
        st.warning("未检测到音频模型文件。")

    st.subheader("歌词模型")
    if lyrics_config:
        st.write(f"预训练模型: {lyrics_config.get('pretrained_model_name', 'bert-base-uncased')}")
        st.write(f"最大长度: {lyrics_config.get('max_length', 128)}")
        if lyrics_config.get('missing_labels'):
            st.write(f"跳过类别: {lyrics_config.get('missing_labels')}")
    else:
        st.warning("未检测到歌词模型文件。")

    uploaded_file = st.file_uploader("上传音乐文件", type=["mp3", "wav", "ogg", "flac", "au", "json"])

st.markdown("<h2 style='text-align: center;'>音乐流派分类 BY CNN + Lyrics BERT</h2>", unsafe_allow_html=True)

audio_json_data = None
audio_data = None
if uploaded_file is not None:
    if uploaded_file.name.endswith('.json'):
        try:
            audio_json_data = json.load(uploaded_file)
        except json.JSONDecodeError:
            st.error("错误: 无法解析上传的 JSON 文件内容。")
    else:
        audio_data = uploaded_file

if audio_json_data is None:
    audio_json_data = load_training_json(audio_config, "training_output.json")
lyrics_json_data = load_training_json(lyrics_config, "lyrics_training_output.json")

col1, col2 = st.columns([3, 2])

with col1:
    if audio_data is not None and audio_model is not None:
        try:
            audio_bytes = audio_data.read()
            wav_file_path = convert_audio_to_wav(audio_bytes)
            audio, sr = librosa.load(wav_file_path)
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

            fig, ax = plt.subplots(figsize=(7, 3.5))
            librosa.display.specshow(mel_spec_db, x_axis='time', y_axis='mel', sr=sr, ax=ax)
            ax.set_title('Mel Spectrogram')
            ax.set_aspect('auto', adjustable='box')
            st.pyplot(fig)

            with open(wav_file_path, 'rb') as f:
                st.audio(f.read(), format="audio/wav")

            predicted_class, probabilities = preprocess_and_predict(
                audio_model,
                wav_file_path,
                target_sr=audio_config["target_sr"],
                n_mfcc=audio_config["n_mfcc"],
                n_mels=audio_config["n_mels"],
                max_length=audio_config["max_length"],
                feature_type=audio_config.get("feature_type", "mfcc"),
                model_type=audio_config.get("model_type", "single"),
                standardize=audio_config.get("standardize", False),
            )
            if predicted_class is not None and probabilities is not None:
                predicted_label = audio_label_mapper.get_label(predicted_class)
                st.success(f"🎵 模型：**{audio_config.get('model_type', 'single')}**，预测音乐风格：**{predicted_label}**")
                display_genre_probabilities(probabilities, audio_label_mapper)
            else:
                st.error("预测时返回了无效结果。")
        except Exception as e:
            st.error(f"音频处理或预测时出现错误: {e}")
    elif audio_data is None:
        st.info("请上传一个音频文件以查看 Mel 频谱图和音频分类结果。")
    else:
        st.warning("未检测到音频模型，无法进行音频分类。")

    st.markdown("---")
    st.subheader("歌词风格分类")

    # ✅ 修改开始：改为上传txt文件
    uploaded_lyrics = st.file_uploader(
        "上传歌词文件（txt）",
        type=["txt"],
        key="lyrics_uploader"
    )

    lyrics_text = ""
    if uploaded_lyrics is not None:
        try:
            lyrics_text = uploaded_lyrics.read().decode("utf-8")
            st.text_area("歌词内容预览", lyrics_text, height=200)
        except Exception as e:
            st.error(f"读取歌词文件失败: {e}")
    # ✅ 修改结束

    if st.button("开始歌词分类"):
        if lyrics_model is None:
            st.error("未检测到歌词模型文件，请先运行 lyrics_train.py。")
        elif not lyrics_text.strip():
            st.warning("请上传歌词文件。")
        else:
            try:
                predicted_class, probabilities, diagnostics = predict_lyrics(
                    lyrics_model,
                    lyrics_text,
                    pretrained_model_name=lyrics_config.get("pretrained_model_name", "bert-base-uncased"),
                    max_length=lyrics_config.get("max_length", 128),
                )
                predicted_label = lyrics_label_mapper.get_label(predicted_class)
                st.success(f"📝 Lyrics BERT 预测风格：**{predicted_label}**")
                display_genre_probabilities(probabilities, lyrics_label_mapper)
                st.json(diagnostics)
            except Exception as e:
                st.error(f"歌词预测时出现错误: {e}")

with col2:
    st.subheader("音频训练曲线")
    if audio_json_data:
        audio_epochs = [d["epoch"] for d in audio_json_data]
        train_loss_fig = plot_graph(audio_epochs, [d["train_loss"] for d in audio_json_data], 'Train Loss', 'Epoch', 'Loss', 'r-')
        train_accuracy_fig = plot_graph(audio_epochs, [d["train_accuracy"] for d in audio_json_data], 'Train Accuracy', 'Epoch', 'Accuracy (%)', 'g-')
        val_loss_fig = plot_graph(audio_epochs, [d["val_loss"] for d in audio_json_data], 'Validation Loss', 'Epoch', 'Loss', 'b-')
        val_accuracy_fig = plot_graph(audio_epochs, [d["val_accuracy"] for d in audio_json_data], 'Validation Accuracy', 'Epoch', 'Accuracy (%)', 'y-')
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
    else:
        st.info("暂无音频训练记录。")

    st.subheader("歌词训练曲线")
    if lyrics_json_data:
        lyrics_epochs = [d["epoch"] for d in lyrics_json_data]
        lyrics_train_loss_fig = plot_graph(lyrics_epochs, [d["train_loss"] for d in lyrics_json_data], 'Lyrics Train Loss', 'Epoch', 'Loss', 'r-')
        lyrics_train_accuracy_fig = plot_graph(lyrics_epochs, [d["train_accuracy"] for d in lyrics_json_data], 'Lyrics Train Accuracy', 'Epoch', 'Accuracy (%)', 'g-')
        lyrics_val_loss_fig = plot_graph(lyrics_epochs, [d["val_loss"] for d in lyrics_json_data], 'Lyrics Val Loss', 'Epoch', 'Loss', 'b-')
        lyrics_val_accuracy_fig = plot_graph(lyrics_epochs, [d["val_accuracy"] for d in lyrics_json_data], 'Lyrics Val Accuracy', 'Epoch', 'Accuracy (%)', 'y-')
        lyrics_val_recall_fig = plot_graph(lyrics_epochs, [d.get("val_recall", 0.0) for d in lyrics_json_data], 'Lyrics Val Recall', 'Epoch', 'Recall (%)', 'm-')
        sub_col5, sub_col6 = st.columns(2)
        with sub_col5:
            st.pyplot(lyrics_train_loss_fig)
        with sub_col6:
            st.pyplot(lyrics_train_accuracy_fig)
        sub_col7, sub_col8 = st.columns(2)
        with sub_col7:
            st.pyplot(lyrics_val_loss_fig)
        with sub_col8:
            st.pyplot(lyrics_val_accuracy_fig)
        st.pyplot(lyrics_val_recall_fig)
    else:
        st.info("暂无歌词训练记录。")