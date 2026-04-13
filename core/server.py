import io
import json
import sys
import tempfile
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import streamlit as st
import torch
from transformers import AutoTokenizer

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from feature_utils import extract_dual_branch_features
from fusion_models import build_fusion_model
from lyrics_data_process import compute_repetition_score
from model_factory import load_model_and_config
from pre_process import predict_lyrics, preprocess_and_predict

AUDIO_CONFIG_PATH = Path("best_model_config.json")
AUDIO_MODEL_PATH = Path("best_model.pth")
LYRICS_CONFIG_PATH = Path("lyrics_best_model_config.json")
LYRICS_MODEL_PATH = Path("lyrics_best_model.pth")
MULTIMODAL_MODEL_PATH = Path("multimodal_best_model_dynamic.pth")

# 已保存好的训练曲线图（按你的本地路径）
AUDIO_CURVE_IMAGE_PATH = Path(r"D:\music_classify_project\core\音频通道消融实验结果（不同学习率）\audio_training_metrics_lr_5e-05.png")
LYRICS_CURVE_IMAGE_PATH = Path(r"D:\music_classify_project\core\歌词通道消融实验结果（不同学习率）\lyrics_training_metrics_lr_5e-04.png")
MULTIMODAL_CURVE_IMAGE_PATH = Path(r"D:\music_classify_project\core\\多模态分类消融结果（不同特征策略对比）\multimodal_training_report_dynamic.png")


def load_bundle(config_path, model_path):
    if config_path.exists() and model_path.exists():
        return load_model_and_config(str(config_path), str(model_path))
    return None, {}, None, None


audio_model, audio_config, _, audio_label_mapper = load_bundle(AUDIO_CONFIG_PATH, AUDIO_MODEL_PATH)
lyrics_model, lyrics_config, _, lyrics_label_mapper = load_bundle(LYRICS_CONFIG_PATH, LYRICS_MODEL_PATH)


def load_multimodal_fusion_head():
    if (
        audio_model is None
        or lyrics_model is None
        or not MULTIMODAL_MODEL_PATH.exists()
    ):
        return None

    audio_dim = getattr(audio_model.classifier, "in_features", 128)
    lyrics_dim = getattr(lyrics_model.classifier, "in_features", 256)
    num_classes = getattr(audio_model.classifier, "out_features", 10)
    fusion_head = build_fusion_model(
        fusion="dynamic",
        audio_dim=audio_dim,
        lyrics_dim=lyrics_dim,
        num_classes=num_classes,
    )

    checkpoint = torch.load(str(MULTIMODAL_MODEL_PATH), map_location="cpu")
    if isinstance(checkpoint, dict) and "fusion" in checkpoint:
        fusion_head.load_state_dict(checkpoint["fusion"])
    else:
        fusion_head.load_state_dict(checkpoint)

    device = next(audio_model.parameters()).device
    fusion_head.to(device)
    fusion_head.eval()
    return fusion_head


multimodal_fusion_head = load_multimodal_fusion_head()
_TOKENIZER = AutoTokenizer.from_pretrained(lyrics_config.get("pretrained_model_name", "bert-base-uncased")) if lyrics_config else None


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


def _align_probabilities(probabilities, label_mapper, target_labels):
    if len(probabilities) == len(target_labels):
        return np.asarray(probabilities, dtype=np.float32)

    aligned = np.zeros(len(target_labels), dtype=np.float32)
    if label_mapper is not None:
        source_labels = [str(x).lower() for x in label_mapper.get_labels()]
    else:
        source_labels = [str(i) for i in range(len(probabilities))]
    target_map = {label.lower(): idx for idx, label in enumerate(target_labels)}

    mapped = 0
    for i, label in enumerate(source_labels[: len(probabilities)]):
        if label in target_map:
            aligned[target_map[label]] = probabilities[i]
            mapped += 1
    if mapped == 0:
        aligned[: len(probabilities)] = probabilities
    return aligned


def predict_multimodal(audio_bytes, lyrics_text):
    if audio_model is None or lyrics_model is None or multimodal_fusion_head is None:
        return None, None, None

    wav_file_path = convert_audio_to_wav(audio_bytes)
    audio, sr = librosa.load(wav_file_path, sr=audio_config.get("target_sr", 22050))
    mfcc, mel = extract_dual_branch_features(
        audio,
        sr,
        n_mfcc=audio_config.get("n_mfcc", 13),
        n_mels=audio_config.get("n_mels", 128),
        max_length=audio_config.get("max_length", 1000),
        standardize=audio_config.get("standardize", True),
    )

    device = next(audio_model.parameters()).device
    mfcc = torch.tensor(np.expand_dims(mfcc, axis=-1), dtype=torch.float32).unsqueeze(0).to(device)
    mel = torch.tensor(np.expand_dims(mel, axis=-1), dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        h_mfcc = audio_model.mfcc_branch(mfcc)
        h_mel = audio_model.mel_branch(mel)
        if audio_model.fusion_type == "concat":
            fused = torch.cat([h_mfcc, h_mel], dim=1)
        else:
            gate = audio_model.gate(torch.cat([h_mfcc, h_mel], dim=1))
            fused = gate * h_mfcc + (1 - gate) * h_mel
        z_audio = torch.nn.functional.mish(audio_model.bn_fc1(audio_model.fusion_fc1(fused)))
        audio_logits = audio_model.classifier(audio_model.dropout1(z_audio))
        p_audio = torch.softmax(audio_logits, dim=1)

        if lyrics_text.strip():
            encoded = _TOKENIZER(
                lyrics_text,
                max_length=lyrics_config.get("max_length", 128),
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            repetition_score = torch.tensor([compute_repetition_score(lyrics_text)], dtype=torch.float32, device=device)

            outputs = lyrics_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                repetition_score=repetition_score,
            )
            cls_emb = outputs["cls_embedding"]
            context_emb = outputs["context_embedding"]
            gate = outputs["gate"]
            z_lyrics = lyrics_model.fusion(torch.cat([cls_emb * gate[:, :1], context_emb * gate[:, 1:]], dim=1))
            lyrics_logits = outputs["logits"]
            p_lyrics = torch.softmax(lyrics_logits, dim=1)
            has_lyrics = torch.ones((1, 1), dtype=torch.float32, device=device)
        else:
            z_lyrics = torch.zeros((1, getattr(lyrics_model.classifier, "in_features", 256)), dtype=torch.float32, device=device)
            p_lyrics = torch.zeros_like(p_audio)
            has_lyrics = torch.zeros((1, 1), dtype=torch.float32, device=device)

        target_labels = audio_label_mapper.get_labels() if audio_label_mapper is not None else [str(i) for i in range(p_audio.shape[1])]
        pa = _align_probabilities(p_audio.squeeze(0).cpu().numpy(), audio_label_mapper, target_labels)
        pl = _align_probabilities(
            p_lyrics.squeeze(0).cpu().numpy(),
            lyrics_label_mapper,
            target_labels,
        )
        p_audio = torch.tensor(pa, dtype=torch.float32, device=device).unsqueeze(0)
        p_lyrics = torch.tensor(pl, dtype=torch.float32, device=device).unsqueeze(0)

        fusion_outputs = multimodal_fusion_head(
            z_audio,
            z_lyrics,
            p_audio,
            p_lyrics,
            has_lyrics=has_lyrics,
        )
        probs = fusion_outputs["probabilities"].squeeze(0).cpu().numpy()
        pred_idx = int(np.argmax(probs))
        pred_label = target_labels[pred_idx] if pred_idx < len(target_labels) else str(pred_idx)
        weight = fusion_outputs.get("weights")
        weight_value = float(weight.squeeze().item()) if weight is not None else None
        return pred_label, probs, weight_value


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

    st.subheader("多模态模型")
    if multimodal_fusion_head is not None:
        st.write("融合方式: dynamic")
        st.write(f"融合权重文件: {MULTIMODAL_MODEL_PATH.name}")
    else:
        st.warning("未检测到多模态融合模型文件。")

st.markdown("<h2 style='text-align: center;'>音乐流派分类 BY CNN + Lyrics BERT</h2>", unsafe_allow_html=True)

col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("音频风格分类")
    uploaded_file = st.file_uploader(
        "上传音频文件（mp3/wav/ogg/flac/au）",
        type=["mp3", "wav", "ogg", "flac", "au", "json"],
        key="audio_main_uploader",
    )

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

    st.markdown("---")
    st.subheader("多模态风格分类")

    multimodal_audio_file = st.file_uploader(
        "上传多模态音频文件（mp3/wav/ogg/flac/au）",
        type=["mp3", "wav", "ogg", "flac", "au"],
        key="multimodal_audio_uploader",
    )
    multimodal_lyrics_file = st.file_uploader(
        "上传多模态歌词文件（txt）",
        type=["txt"],
        key="multimodal_lyrics_uploader",
    )

    multimodal_lyrics_text = ""
    if multimodal_lyrics_file is not None:
        try:
            multimodal_lyrics_text = multimodal_lyrics_file.read().decode("utf-8")
            st.text_area("多模态歌词内容预览", multimodal_lyrics_text, height=160, key="multimodal_lyrics_preview")
        except Exception as e:
            st.error(f"读取多模态歌词文件失败: {e}")

    if st.button("开始多模态分类"):
        if multimodal_audio_file is None:
            st.warning("请先上传多模态音频文件。")
        elif multimodal_fusion_head is None:
            st.warning("未检测到多模态融合模型文件，请先训练并放置 multimodal_best_model_dynamic.pth。")
        else:
            try:
                audio_bytes = multimodal_audio_file.getvalue()
                pred_label, probs, weight = predict_multimodal(audio_bytes, multimodal_lyrics_text)
                if pred_label is None:
                    st.error("多模态预测失败，请检查模型与依赖。")
                else:
                    st.success(f"🎯 多模态预测风格：**{pred_label}**")
                    if weight is not None:
                        st.info(f"动态融合权重 w（音频占比）: {weight:.4f}")
                    display_genre_probabilities(probs, audio_label_mapper)
            except Exception as e:
                st.error(f"多模态预测时出现错误: {e}")

with col2:
    st.subheader("音频训练曲线")
    if AUDIO_CURVE_IMAGE_PATH.exists():
        st.image(str(AUDIO_CURVE_IMAGE_PATH), use_container_width=True, caption=AUDIO_CURVE_IMAGE_PATH.name)
    else:
        st.warning(f"未找到音频训练曲线图片：{AUDIO_CURVE_IMAGE_PATH}")

    st.subheader("歌词训练曲线")
    if LYRICS_CURVE_IMAGE_PATH.exists():
        st.image(str(LYRICS_CURVE_IMAGE_PATH), use_container_width=True, caption=LYRICS_CURVE_IMAGE_PATH.name)
    else:
        st.warning(f"未找到歌词训练曲线图片：{LYRICS_CURVE_IMAGE_PATH}")

    st.subheader("多模态训练结果")
    if MULTIMODAL_CURVE_IMAGE_PATH.exists():
        st.image(str(MULTIMODAL_CURVE_IMAGE_PATH), use_container_width=True, caption=MULTIMODAL_CURVE_IMAGE_PATH.name)
    else:
        st.warning(f"未找到多模态训练结果图片：{MULTIMODAL_CURVE_IMAGE_PATH}")