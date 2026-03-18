import io
import tempfile

import torch
import json
import librosa
import librosa.display
import numpy as np
import soundfile as sf
import streamlit as st
import matplotlib.pyplot as plt  # 确保导入这个库

from cnn import AudioCNN
from label_mapper import GTZANLabelMapper
from pre_process import preprocess_and_predict

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
model = AudioCNN()  # 确保使用与你训练时相同的模型结构
model.load_state_dict(torch.load('best_model_test.pth', map_location=device))
model.to(device)  # 将模型移动到相应的设备上
model.eval()  # 切换到评估模式

# 侧边栏菜单
with st.sidebar:
    st.title("菜单")
    uploaded_file = st.file_uploader("上传音乐文件", type=["mp3", "wav", "ogg", "flac", "au"])

# 显示标题，调整字体大小
st.markdown("<h2 style='text-align: center;'>音乐流派分类 BY CNN:ACC >=87.5%</h2>", unsafe_allow_html=True)

# 处理上传文件
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

# 若未上传 JSON 文件，则尝试读取本地文件
if json_data is None:
    try:
        with open('training_output.json', 'r') as file:
            json_data = json.load(file)
    except FileNotFoundError:
        st.error("错误: 未找到 'training_output.json' 文件。")
    except json.JSONDecodeError:
        st.error("错误: 无法解析 'training_output.json' 文件的 JSON 内容。")

# 读取 .au 文件并保存为 .wav 格式
def convert_au_to_wav_librosa(audio_data):
    # 使用 librosa 加载 .au 文件
    audio, sr = librosa.load(io.BytesIO(audio_data), sr=None)

    # 保存为临时 .wav 文件
    wav_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(wav_file.name, audio, sr)
    return wav_file.name

# 显示类别及其概率（2列展示，每列5个类别）
def display_genre_probabilities(probabilities):
    categories = [
        '布鲁斯', '古典', '乡村', '迪斯科', '嘻哈',
        '爵士', '金属', '流行', '雷鬼', '摇滚'
    ]
    # 显示每个类别的预测概率

    # 将每列的类别分开，每列显示5个类别
    col1, col2 = st.columns(2)  # 创建两列布局
    with col1:
        for i in range(5):
            prob_percent = probabilities[i] * 100  # 转换为百分比
            st.write(f"{categories[i]} - {prob_percent:.2f}%")  # 显示前5个类别
    with col2:
        for i in range(5, 10):
            prob_percent = probabilities[i] * 100  # 转换为百分比
            st.write(f"{categories[i]} - {prob_percent:.2f}%")  # 显示后5个类别

if json_data is not None:
    # 提取 epoch、train_loss、train_accuracy、val_loss、val_accuracy
    epochs = [d["epoch"] for d in json_data]
    train_loss = [d["train_loss"] for d in json_data]
    train_accuracy = [d["train_accuracy"] for d in json_data]
    val_loss = [d["val_loss"] for d in json_data]
    val_accuracy = [d["val_accuracy"] for d in json_data]

    # 定义绘图函数，缩小图表尺寸
    def plot_graph(epochs, data, title, x_label, y_label, color):
        fig = plt.figure(figsize=(2, 2))  # 修改图表尺寸
        plt.plot(epochs, data, color)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        return fig

    # 绘制四张图片
    train_loss_fig = plot_graph(epochs, train_loss, 'Train Loss', 'Epoch', 'Loss', 'r-')
    train_accuracy_fig = plot_graph(epochs, train_accuracy, 'Train Accuracy', 'Epoch', 'Accuracy (%)', 'g-')
    val_loss_fig = plot_graph(epochs, val_loss, 'Validation Loss', 'Epoch', 'Loss', 'b-')
    val_accuracy_fig = plot_graph(epochs, val_accuracy, 'Validation Accuracy', 'Epoch', 'Accuracy (%)', 'y-')

    # 创建两列布局，左侧显示音乐播放窗口和 Mel 图谱，右侧显示四宫格图
    col1, col2 = st.columns([3, 2])

    with col1:
        # 读取音频文件并转换为 .wav 格式
        if audio_data is not None:
            try:
                # 读取音频文件并转换为 .wav 格式
                audio_bytes = audio_data.read()
                wav_file_path = convert_au_to_wav_librosa(audio_bytes)

                # 使用 librosa 计算 Mel 频谱
                audio, sr = librosa.load(wav_file_path)

                mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

                # 获取 Mel 频谱的尺寸
                height, width = mel_spec_db.shape  # height=频率bins，width=时间步数

                # 调整图像大小：保证宽度与左侧图表一致，设定一个固定的高度
                fig, ax = plt.subplots(figsize=(6, 3))  # 调整为较窄的图像
                librosa.display.specshow(mel_spec_db, x_axis='time', y_axis='mel', sr=sr, ax=ax)
                ax.set_title('Mel Spectrogram')

                # 强制图像纵横比，避免过度压缩或拉伸
                ax.set_aspect('auto', adjustable='box')

                # 展示 Mel 频谱图
                st.pyplot(fig)

                # 显示音频播放器
                with open(wav_file_path, 'rb') as f:
                    st.audio(f.read(), format="audio/wav")
            except Exception as e:
                st.error(f"播放音频或生成 Mel 频谱图时出现错误: {e}")
        else:
            st.error("请上传一个音频文件以查看 Mel 频谱图。")

        # 预测流派并显示预测概率
        if audio_data is not None:
            try:
                # 使用 GTZANLabelMapper 进行标签映射
                label_mapper = GTZANLabelMapper()

                # 调用函数进行预测
                predicted_class, probabilities = preprocess_and_predict(model, wav_file_path)

                if predicted_class is not None and probabilities is not None:
                    # 获取预测的中文标签
                    predicted_label = label_mapper.get_label(predicted_class)

                    # 显示预测结果弹窗
                    st.success(f"🎵 预测音乐风格：**{predicted_label}**")

                    # 显示各流派的预测概率
                    display_genre_probabilities(probabilities)

                else:
                    st.error("预测时返回了无效结果。")

            except Exception as e:
                st.error(f"预测时出现错误: {e}")

    with col2:
        # 创建两行两列布局显示四张图片
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
