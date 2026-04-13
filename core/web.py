import base64
import io
import json
import os
import random
import sys
import tempfile
from pathlib import Path

import librosa
import numpy as np
import requests
import soundfile as sf
import torch
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func, or_
from transformers import AutoTokenizer
from werkzeug.security import check_password_hash, generate_password_hash

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from feature_utils import extract_dual_branch_features
from fusion_models import build_fusion_model
from lyrics_data_process import compute_repetition_score
from model_factory import load_model_and_config
from pre_process import predict_lyrics, preprocess_and_predict_file

AUDIO_CONFIG_PATH = Path("best_model_config.json")
AUDIO_MODEL_PATH = Path("best_model.pth")
LYRICS_CONFIG_PATH = Path("lyrics_best_model_config.json")
LYRICS_MODEL_PATH = Path("lyrics_best_model.pth")
MULTIMODAL_MODEL_PATH = Path("multimodal_best_model_dynamic.pth")

DEFAULT_DATABASE_URI = os.getenv(
    "DATABASE_URI",
    "mysql+pymysql://root:Wuyipeng427@127.0.0.1:3306/music_classify",
)

GENRE_KEYS = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock",
]

GENRE_STORAGE_KEYS = [f"genre_{label}" for label in GENRE_KEYS]


def load_inference_bundle(config_path, model_path):
    if not config_path.exists() or not model_path.exists():
        return {"model": None, "config": {}, "device": None, "label_mapper": None}
    model, config, device, label_mapper = load_model_and_config(str(config_path), str(model_path))
    return {
        "model": model,
        "config": config,
        "device": device,
        "label_mapper": label_mapper,
    }


AUDIO_BUNDLE = load_inference_bundle(AUDIO_CONFIG_PATH, AUDIO_MODEL_PATH)
LYRICS_BUNDLE = load_inference_bundle(LYRICS_CONFIG_PATH, LYRICS_MODEL_PATH)

app = Flask(__name__)
CORS(app)

app.config["SQLALCHEMY_DATABASE_URI"] = DEFAULT_DATABASE_URI
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(255), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    avatar = db.Column(db.String(255), nullable=True)


class Music(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    song_name = db.Column(db.String(255), nullable=False)
    singer_name = db.Column(db.String(255), nullable=False)
    user_id = db.Column(db.Integer, nullable=False)
    music_file = db.Column(db.Text, nullable=False)
    face_file = db.Column(db.Text, nullable=False)
    genre = db.Column(db.String(255), nullable=False)
    genre_blues = db.Column(db.Float, nullable=False, default=0.0)
    genre_classical = db.Column(db.Float, nullable=False, default=0.0)
    genre_country = db.Column(db.Float, nullable=False, default=0.0)
    genre_disco = db.Column(db.Float, nullable=False, default=0.0)
    genre_hiphop = db.Column(db.Float, nullable=False, default=0.0)
    genre_jazz = db.Column(db.Float, nullable=False, default=0.0)
    genre_metal = db.Column(db.Float, nullable=False, default=0.0)
    genre_pop = db.Column(db.Float, nullable=False, default=0.0)
    genre_reggae = db.Column(db.Float, nullable=False, default=0.0)
    genre_rock = db.Column(db.Float, nullable=False, default=0.0)


class Collection(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    music_id = db.Column(db.Integer, nullable=False)
    __table_args__ = (db.UniqueConstraint("user_id", "music_id", name="unique_user_music"),)


with app.app_context():
    db.create_all()


def generate_random_image():
    index = random.randint(1, 1000)
    response = requests.get(f"https://picsum.photos/200/200?random={index}", timeout=10)
    if response.status_code == 200:
        return base64.b64encode(response.content).decode("utf-8")
    return ""


def decode_base64_audio(audio_base64):
    if not audio_base64:
        return None
    if "base64," in audio_base64:
        audio_base64 = audio_base64.split("base64,", 1)[1]
    return base64.b64decode(audio_base64)


def probabilities_to_response(probabilities):
    if probabilities is None:
        return {f"genre_{label}": 0.0 for label in GENRE_KEYS}
    response = {f"genre_{label}": 0.0 for label in GENRE_KEYS}
    for i in range(min(len(GENRE_KEYS), len(probabilities))):
        response[f"genre_{GENRE_KEYS[i]}"] = float(probabilities[i])
    return response


def get_training_metrics(config):
    training_path = config.get("training_output_path", "training_output.json")
    file_path = Path(training_path)
    if not file_path.exists():
        return []
    try:
        return json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def netease_search_song(keyword, limit=10):
    if not keyword:
        return []
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://music.163.com/"}
    search_url = "https://music.163.com/api/search/get"
    lyric_url = "https://music.163.com/api/song/lyric"

    try:
        resp = requests.get(
            search_url,
            params={"s": keyword, "type": 1, "limit": limit},
            headers=headers,
            timeout=10,
        )
        payload = resp.json()
    except Exception:
        return []

    songs = payload.get("result", {}).get("songs", []) or []
    items = []
    for song in songs:
        song_id = song.get("id")
        if not song_id:
            continue

        song_name = song.get("name", "")
        artists = ",".join([a.get("name", "") for a in song.get("artists", []) if a.get("name")])
        album_pic = (song.get("album") or {}).get("picUrl", "")
        audio_url = f"http://music.163.com/song/media/outer/url?id={song_id}.mp3"
        song_page = f"http://music.163.com/#/song?id={song_id}"
        iframe_url = f"//music.163.com/outchain/player?type=2&id={song_id}&auto=0&height=66"

        lyric_text = ""
        try:
            lr = requests.get(
                lyric_url,
                params={"id": song_id, "lv": -1, "tv": -1},
                headers=headers,
                timeout=8,
            )
            lyric_payload = lr.json()
            lyric_text = (lyric_payload.get("lrc") or {}).get("lyric", "") or ""
        except Exception:
            lyric_text = ""

        lyric_preview = (lyric_text.strip()[:180] + " ...") if lyric_text.strip() else "无歌词"

        items.append(
            {
                "song_id": song_id,
                "song_name": song_name,
                "artists": artists,
                "song_page": song_page,
                "audio_url": audio_url,
                "audio_proxy_url": f"/music_audio_proxy?id={song_id}",
                "lyric_text": lyric_text,
                "lyric_preview": lyric_preview,
                "album_pic": album_pic,
                "iframe_html": f'<iframe frameborder="no" border="0" marginwidth="0" marginheight="0" width="330" height="86" src="{iframe_url}"></iframe>',
            }
        )
    return items


def _music_to_dict(music):
    return {
        "id": music.id,
        "song_name": music.song_name,
        "singer_name": music.singer_name,
        "face_file": music.face_file,
        "genre": music.genre,
        "genreProbabilities": {
            "genre_blues": float(music.genre_blues),
            "genre_classical": float(music.genre_classical),
            "genre_country": float(music.genre_country),
            "genre_disco": float(music.genre_disco),
            "genre_hiphop": float(music.genre_hiphop),
            "genre_jazz": float(music.genre_jazz),
            "genre_metal": float(music.genre_metal),
            "genre_pop": float(music.genre_pop),
            "genre_reggae": float(music.genre_reggae),
            "genre_rock": float(music.genre_rock),
        },
    }


def verify_password(stored_password, candidate_password):
    if stored_password == candidate_password:
        return True
    return check_password_hash(stored_password, candidate_password)


def load_multimodal_fusion_head():
    if (
        AUDIO_BUNDLE["model"] is None
        or LYRICS_BUNDLE["model"] is None
        or not MULTIMODAL_MODEL_PATH.exists()
    ):
        return None

    audio_model = AUDIO_BUNDLE["model"]
    lyrics_model = LYRICS_BUNDLE["model"]
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


MULTIMODAL_FUSION_HEAD = load_multimodal_fusion_head()
TOKENIZER = None
if LYRICS_BUNDLE["config"]:
    try:
        TOKENIZER = AutoTokenizer.from_pretrained(
            LYRICS_BUNDLE["config"].get("pretrained_model_name", "bert-base-uncased")
        )
    except Exception:
        TOKENIZER = None


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


def predict_multimodal_from_bytes(audio_bytes, lyrics_text):
    if (
        AUDIO_BUNDLE["model"] is None
        or LYRICS_BUNDLE["model"] is None
        or MULTIMODAL_FUSION_HEAD is None
        or TOKENIZER is None
    ):
        return None, None, None

    wav_file_path = convert_audio_to_wav(audio_bytes)
    try:
        audio, sr = librosa.load(wav_file_path, sr=AUDIO_BUNDLE["config"].get("target_sr", 22050))
        mfcc, mel = extract_dual_branch_features(
            audio,
            sr,
            n_mfcc=AUDIO_BUNDLE["config"].get("n_mfcc", 13),
            n_mels=AUDIO_BUNDLE["config"].get("n_mels", 128),
            max_length=AUDIO_BUNDLE["config"].get("max_length", 1000),
            standardize=AUDIO_BUNDLE["config"].get("standardize", True),
        )

        audio_model = AUDIO_BUNDLE["model"]
        lyrics_model = LYRICS_BUNDLE["model"]
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
                encoded = TOKENIZER(
                    lyrics_text,
                    max_length=LYRICS_BUNDLE["config"].get("max_length", 128),
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                input_ids = encoded["input_ids"].to(device)
                attention_mask = encoded["attention_mask"].to(device)
                repetition_score = torch.tensor(
                    [compute_repetition_score(lyrics_text)],
                    dtype=torch.float32,
                    device=device,
                )

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
                z_lyrics = torch.zeros(
                    (1, getattr(lyrics_model.classifier, "in_features", 256)),
                    dtype=torch.float32,
                    device=device,
                )
                p_lyrics = torch.zeros_like(p_audio)
                has_lyrics = torch.zeros((1, 1), dtype=torch.float32, device=device)

            target_labels = (
                AUDIO_BUNDLE["label_mapper"].get_labels()
                if AUDIO_BUNDLE["label_mapper"] is not None
                else [str(i) for i in range(p_audio.shape[1])]
            )
            pa = _align_probabilities(p_audio.squeeze(0).cpu().numpy(), AUDIO_BUNDLE["label_mapper"], target_labels)
            pl = _align_probabilities(p_lyrics.squeeze(0).cpu().numpy(), LYRICS_BUNDLE["label_mapper"], target_labels)
            p_audio = torch.tensor(pa, dtype=torch.float32, device=device).unsqueeze(0)
            p_lyrics = torch.tensor(pl, dtype=torch.float32, device=device).unsqueeze(0)

            fusion_outputs = MULTIMODAL_FUSION_HEAD(
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
    finally:
        try:
            os.unlink(wav_file_path)
        except OSError:
            pass


@app.route("/")
def index():
    return """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI音乐风格分类系统</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: "Segoe UI", Arial, sans-serif;
  color: #eaf2ff;
  background: radial-gradient(circle at 20% 20%, #233a63, #121a2e 55%, #0d1424);
}
.top-nav {
  position: sticky;
  top: 0;
  z-index: 10;
  display: flex;
  gap: 4px;
  background: rgba(10, 18, 32, 0.95);
  border-bottom: 1px solid rgba(255,255,255,0.15);
  padding: 8px 10px;
}
.nav-btn {
  background: transparent;
  border: 1px solid transparent;
  color: #dbe9ff;
  padding: 10px 14px;
}
.nav-btn.active {
  background: rgba(61, 120, 255, 0.25);
  border-color: rgba(97, 154, 255, 0.7);
}
.page {
  max-width: 1200px;
  margin: 0 auto;
  padding: 24px 16px 40px;
}
h1 {
  margin: 0 0 8px 0;
  text-align: center;
  letter-spacing: 0.5px;
}
.subtitle {
  text-align: center;
  color: #9cb3d4;
  margin-bottom: 24px;
}
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
  gap: 16px;
}
.card {
  background: rgba(255, 255, 255, 0.08);
  border: 1px solid rgba(255, 255, 255, 0.16);
  border-radius: 16px;
  padding: 16px;
  box-shadow: 0 12px 28px rgba(0, 0, 0, 0.28);
}
.card h2 { margin: 0 0 12px; font-size: 20px; }
.row { margin-bottom: 10px; }
.row label {
  display: block;
  margin-bottom: 4px;
  font-size: 13px;
  color: #b7c9e7;
}
input, textarea {
  width: 100%;
  border: 1px solid #3b4e6e;
  border-radius: 10px;
  background: #10192b;
  color: #eaf2ff;
  padding: 10px;
}
textarea { min-height: 120px; resize: vertical; }
button {
  border: none;
  border-radius: 10px;
  background: linear-gradient(90deg, #2dd4ff, #3879ff);
  color: #fff;
  padding: 10px 14px;
  cursor: pointer;
  font-weight: 600;
}
button:hover { filter: brightness(1.08); }
.result {
  margin-top: 10px;
  min-height: 24px;
  color: #d6e6ff;
}
audio {
  width: 100%;
  margin-top: 8px;
}
.metrics {
  margin-top: 8px;
  font-size: 13px;
  color: #9cc3ff;
}
.panel { display: none; }
.panel.active { display: block; }
.manager-layout {
  display: grid;
  grid-template-columns: 360px 1fr;
  gap: 16px;
}
.manager-player-lyrics {
  white-space: pre-wrap;
  line-height: 1.6;
  max-height: 460px;
  overflow-y: auto;
  background: #0f192c;
  border: 1px solid #364b71;
  border-radius: 10px;
  padding: 10px;
}
.genre-tabs {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 10px;
}
.genre-tab {
  padding: 6px 10px;
  border-radius: 8px;
  border: 1px solid #4c648d;
  background: #101b30;
  color: #d8e7ff;
  cursor: pointer;
}
.genre-tab.active {
  background: #2d62c2;
  border-color: #74a5ff;
}
table {
  width: 100%;
  border-collapse: collapse;
}
th, td {
  border-bottom: 1px solid rgba(255,255,255,0.12);
  text-align: left;
  padding: 8px 6px;
  font-size: 14px;
}
.danger-btn {
  background: linear-gradient(90deg, #ff6b6b, #d64545);
}
@media (max-width: 980px){
  .manager-layout { grid-template-columns: 1fr; }
}
</style>
</head>
<body>
<div class="top-nav">
  <button id="tabClassifyBtn" class="nav-btn active" onclick="switchPanel('classify')">音乐风格分类</button>
  <button id="tabSearchBtn" class="nav-btn" onclick="switchPanel('search')">音乐搜索</button>
  <button id="tabManagerBtn" class="nav-btn" onclick="switchPanel('manager')">音乐管理</button>
</div>

<div class="page">
  <section id="panelClassify" class="panel active">
    <h1>🎵 AI 音乐风格分类系统</h1>
    <p class="subtitle">音频分类 / 歌词分类 / 多模态融合分类（独立上传框）</p>
    <div class="grid">
      <section class="card">
        <h2>音频分类</h2>
        <div class="row"><label>音频文件（mp3/wav/ogg/flac/au）</label><input type="file" id="musicFile" accept=".mp3,.wav,.ogg,.flac,.au"></div>
        <button onclick="uploadMusic()">上传并分类</button>
        <div class="result" id="genreResult"></div>
        <audio id="audioPlayer" controls></audio>
        <canvas id="audioProbChart"></canvas>
      </section>

      <section class="card">
        <h2>歌词分类</h2>
        <div class="row"><label>歌词文本</label><textarea id="lyricsText" placeholder="在这里输入歌词"></textarea></div>
        <div class="row"><label>或上传歌词文件（txt）</label><input type="file" id="lyricsFile" accept=".txt"></div>
        <button onclick="predictLyrics()">歌词预测</button>
        <div class="result" id="lyricsResult"></div>
        <canvas id="lyricsProbChart"></canvas>
      </section>

      <section class="card">
        <h2>多模态分类（独立输入）</h2>
        <div class="row"><label>多模态音频文件</label><input type="file" id="multimodalMusicFile" accept=".mp3,.wav,.ogg,.flac,.au"></div>
        <div class="row"><label>多模态歌词文本</label><textarea id="multimodalLyricsText" placeholder="可选：输入歌词可提升融合效果"></textarea></div>
        <div class="row"><label>或上传多模态歌词文件（txt）</label><input type="file" id="multimodalLyricsFile" accept=".txt"></div>
        <button onclick="predictMultimodal()">多模态预测</button>
        <div class="result" id="multimodalResult"></div>
        <div class="metrics" id="fusionWeightText"></div>
        <audio id="multimodalAudioPlayer" controls></audio>
        <canvas id="multimodalProbChart"></canvas>
      </section>
    </div>
  </section>

  <section id="panelSearch" class="panel">
    <h1>🔎 音乐搜索器</h1>
    <p class="subtitle">输入歌名进行搜索，返回歌曲信息、音频链接、歌词与可播放外链播放器。</p>
    <section class="card">
      <div class="row"><label>歌曲关键词</label><input id="searchKeyword" placeholder="例如：The Thrill Is Gone B.B. King"></div>
      <button onclick="searchMusicNow()">搜索</button>
      <div class="result" id="searchStatus"></div>
      <div id="searchResults" style="margin-top:12px;"></div>
    </section>
  </section>

  <section id="panelManager" class="panel">
    <h1>🎼 音乐管理</h1>
    <p class="subtitle">管理你的歌曲库：按分类查看、添加、播放、删除，并在播放器中同步显示歌词。</p>
    <div class="manager-layout">
      <section class="card">
        <h2>音乐播放器</h2>
        <div class="result" id="managerNowPlaying">当前未播放</div>
        <audio id="managerAudioPlayer" controls></audio>
        <h3>歌词</h3>
        <div id="managerLyricsDisplay" class="manager-player-lyrics">请选择右侧歌曲进行播放，歌词会显示在这里。</div>
      </section>
      <section class="card">
        <h2>音乐库</h2>
        <div id="managerGenreTabs" class="genre-tabs"></div>
        <table>
          <thead>
            <tr><th>歌名</th><th>分类</th><th>操作</th></tr>
          </thead>
          <tbody id="managerSongTable"></tbody>
        </table>
        <hr style="border-color: rgba(255,255,255,0.18)">
        <h3>添加音乐</h3>
        <div class="row"><label>歌名</label><input id="manageSongName" placeholder="输入歌名"></div>
        <div class="row"><label>分类</label><select id="manageGenreSelect"></select></div>
        <div class="row"><label>音频文件</label><input id="manageAudioFile" type="file" accept=".mp3,.wav,.ogg,.flac,.au"></div>
        <div class="row"><label>歌词文本</label><textarea id="manageLyricsText" placeholder="可直接输入歌词"></textarea></div>
        <div class="row"><label>或上传歌词文件（txt/lrc）</label><input id="manageLyricsFile" type="file" accept=".txt,.lrc"></div>
        <button onclick="addManagedSong()">添加音乐</button>
      </section>
    </div>
  </section>
</div>

<script>
const labels = ["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]
let audioChart = null
let lyricsChart = null
let multimodalChart = null
let managerSelectedGenre = labels[0]
const managedSongs = []

function readTextFile(file){
  return new Promise((resolve,reject)=>{
    const reader = new FileReader()
    reader.onload = () => resolve(reader.result || "")
    reader.onerror = reject
    reader.readAsText(file)
  })
}

function base64FromFile(file){
  return new Promise((resolve,reject)=>{
    const reader = new FileReader()
    reader.onload = () => resolve(reader.result)
    reader.onerror = reject
    reader.readAsDataURL(file)
  })
}

function switchPanel(panel){
  const classifyPanel = document.getElementById("panelClassify")
  const searchPanel = document.getElementById("panelSearch")
  const managerPanel = document.getElementById("panelManager")
  const classifyBtn = document.getElementById("tabClassifyBtn")
  const searchBtn = document.getElementById("tabSearchBtn")
  const managerBtn = document.getElementById("tabManagerBtn")
  classifyPanel.classList.remove("active")
  searchPanel.classList.remove("active")
  managerPanel.classList.remove("active")
  classifyBtn.classList.remove("active")
  searchBtn.classList.remove("active")
  managerBtn.classList.remove("active")
  if(panel === "manager"){
    managerPanel.classList.add("active")
    managerBtn.classList.add("active")
  }else if(panel === "search"){
    searchPanel.classList.add("active")
    searchBtn.classList.add("active")
  }else{
    classifyPanel.classList.add("active")
    classifyBtn.classList.add("active")
  }
}

async function searchMusicNow(){
  const keyword = (document.getElementById("searchKeyword").value || "").trim()
  if (!keyword){ alert("请输入歌曲关键词"); return }
  document.getElementById("searchStatus").innerText = "搜索中..."
  const resp = await fetch(`/music_search_api?keyword=${encodeURIComponent(keyword)}`)
  const data = await resp.json()
  if (!resp.ok){
    document.getElementById("searchStatus").innerText = data.message || "搜索失败"
    return
  }
  const items = data.results || []
  document.getElementById("searchStatus").innerText = `搜索完成，共 ${items.length} 条结果`
  const wrap = document.getElementById("searchResults")
  if (!items.length){
    wrap.innerHTML = "<div class='result'>未找到相关歌曲</div>"
    return
  }
  wrap.innerHTML = items.map((item, idx) => `
    <div class="card" style="margin-bottom:12px;">
      <div><strong>${idx + 1}. ${item.song_name || ""}</strong> - ${item.artists || ""}</div>
      <div class="metrics">Song ID: ${item.song_id || ""}</div>
      <div class="metrics"><a href="${item.song_page}" target="_blank" style="color:#8fc2ff;">歌曲页面链接</a></div>
      <div class="metrics"><a href="${item.audio_url}" target="_blank" style="color:#8fc2ff;">音频链接</a></div>
      <div class="metrics">歌词预览：${item.lyric_preview || "无歌词"}</div>
      <audio controls style="width:330px; margin-top:8px;" src="${item.audio_proxy_url || ""}"></audio>
      <div style="margin-top:8px;">${item.iframe_html || ""}</div>
    </div>
  `).join("")
}

function renderChart(canvasId, probabilities, chartType){
  const values = labels.map(x => Number(probabilities["genre_" + x] || 0))
  if (chartType === "audio" && audioChart){ audioChart.destroy() }
  if (chartType === "lyrics" && lyricsChart){ lyricsChart.destroy() }
  if (chartType === "multimodal" && multimodalChart){ multimodalChart.destroy() }
  const chart = new Chart(document.getElementById(canvasId), {
    type: "bar",
    data: { labels, datasets: [{ label: "概率", data: values }] },
    options: { scales: { y: { beginAtZero: true, max: 1 } } }
  })
  if (chartType === "audio"){ audioChart = chart }
  if (chartType === "lyrics"){ lyricsChart = chart }
  if (chartType === "multimodal"){ multimodalChart = chart }
}

document.getElementById("lyricsFile").addEventListener("change", async (e) => {
  const file = e.target.files[0]
  if (!file) return
  document.getElementById("lyricsText").value = await readTextFile(file)
})

document.getElementById("multimodalLyricsFile").addEventListener("change", async (e) => {
  const file = e.target.files[0]
  if (!file) return
  document.getElementById("multimodalLyricsText").value = await readTextFile(file)
})

document.getElementById("manageLyricsFile").addEventListener("change", async (e) => {
  const file = e.target.files[0]
  if (!file) return
  document.getElementById("manageLyricsText").value = await readTextFile(file)
})

async function uploadMusic(){
  const file = document.getElementById("musicFile").files[0]
  if (!file){ alert("请先选择音频文件"); return }
  const b64 = await base64FromFile(file)
  const payload = {
    songName: "未命名歌曲",
    singerName: "未知歌手",
    userId: 1,
    musicFile: b64
  }
  const resp = await fetch("/upload_music", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  })
  const data = await resp.json()
  if (!resp.ok){ alert(data.message || "上传失败"); return }
  document.getElementById("genreResult").innerText = `音频预测风格：${data.genre}`
  document.getElementById("audioPlayer").src = b64
  renderChart("audioProbChart", data.probabilities || {}, "audio")
}

async function predictLyrics(){
  const lyricsText = (document.getElementById("lyricsText").value || "").trim()
  if (!lyricsText){ alert("请输入歌词文本"); return }
  const resp = await fetch("/predict_lyrics", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ lyrics_text: lyricsText })
  })
  const data = await resp.json()
  if (!resp.ok){ alert(data.message || "歌词预测失败"); return }
  document.getElementById("lyricsResult").innerText = `歌词预测风格：${data.genre}`
  renderChart("lyricsProbChart", data.probabilities || {}, "lyrics")
}

async function predictMultimodal(){
  const file = document.getElementById("multimodalMusicFile").files[0]
  if (!file){ alert("请先选择多模态音频文件"); return }
  const b64 = await base64FromFile(file)
  const lyricsText = (document.getElementById("multimodalLyricsText").value || "").trim()
  const resp = await fetch("/predict_multimodal", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ musicFile: b64, lyrics_text: lyricsText })
  })
  const data = await resp.json()
  if (!resp.ok){ alert(data.message || "多模态预测失败"); return }
  document.getElementById("multimodalAudioPlayer").src = b64
  document.getElementById("multimodalResult").innerText = `多模态预测风格：${data.genre}`
  document.getElementById("fusionWeightText").innerText =
    data.fusion_weight === null || data.fusion_weight === undefined
    ? ""
    : `融合权重：${Number(data.fusion_weight).toFixed(4)}`
  renderChart("multimodalProbChart", data.probabilities || {}, "multimodal")
}

function renderManagerGenreTabs(){
  const wrap = document.getElementById("managerGenreTabs")
  wrap.innerHTML = ""
  labels.forEach((genre) => {
    const btn = document.createElement("button")
    btn.className = "genre-tab" + (managerSelectedGenre === genre ? " active" : "")
    btn.textContent = genre
    btn.onclick = () => {
      managerSelectedGenre = genre
      renderManagerGenreTabs()
      renderManagerSongTable()
    }
    wrap.appendChild(btn)
  })
}

function renderManageGenreSelect(){
  const select = document.getElementById("manageGenreSelect")
  select.innerHTML = labels.map((genre) => `<option value="${genre}">${genre}</option>`).join("")
}

function renderManagerSongTable(){
  const tbody = document.getElementById("managerSongTable")
  const songs = managedSongs.filter((x) => x.genre === managerSelectedGenre)
  if (songs.length === 0){
    tbody.innerHTML = `<tr><td colspan="3">当前分类暂无歌曲</td></tr>`
    return
  }
  tbody.innerHTML = songs.map((song) => `
    <tr>
      <td>${song.name}</td>
      <td>${song.genre}</td>
      <td>
        <button onclick="playManagedSong('${song.id}')">播放</button>
        <button class="danger-btn" onclick="deleteManagedSong('${song.id}')">删除</button>
      </td>
    </tr>
  `).join("")
}

function playManagedSong(songId){
  const song = managedSongs.find((x) => x.id === songId)
  if (!song) return
  document.getElementById("managerNowPlaying").innerText = `正在播放：${song.name} (${song.genre})`
  document.getElementById("managerLyricsDisplay").innerText = song.lyrics || "暂无歌词"
  const audio = document.getElementById("managerAudioPlayer")
  audio.src = song.audioUrl
  audio.play()
}

function deleteManagedSong(songId){
  const idx = managedSongs.findIndex((x) => x.id === songId)
  if (idx >= 0){
    const removed = managedSongs[idx]
    if (removed.audioUrl){
      URL.revokeObjectURL(removed.audioUrl)
    }
    managedSongs.splice(idx, 1)
  }
  renderManagerSongTable()
}

async function addManagedSong(){
  const name = (document.getElementById("manageSongName").value || "").trim()
  const genre = document.getElementById("manageGenreSelect").value
  const audioFile = document.getElementById("manageAudioFile").files[0]
  const lyrics = (document.getElementById("manageLyricsText").value || "").trim()
  if (!name){ alert("请填写歌名"); return }
  if (!audioFile){ alert("请上传音频文件"); return }

  const audioUrl = URL.createObjectURL(audioFile)
  managedSongs.push({
    id: `${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
    name,
    genre,
    lyrics,
    audioUrl,
  })
  document.getElementById("manageSongName").value = ""
  document.getElementById("manageAudioFile").value = ""
  document.getElementById("manageLyricsText").value = ""
  document.getElementById("manageLyricsFile").value = ""
  managerSelectedGenre = genre
  renderManagerGenreTabs()
  renderManagerSongTable()
  alert("添加成功")
}

renderManageGenreSelect()
renderManagerGenreTabs()
renderManagerSongTable()
</script>
</body>
</html>
"""


@app.route("/model_info", methods=["GET"])
def model_info():
    return jsonify({
        "audio_model": AUDIO_BUNDLE["config"],
        "lyrics_model": LYRICS_BUNDLE["config"],
    })


@app.route("/lyrics_training_metrics", methods=["GET"])
def lyrics_training_metrics():
    return jsonify(get_training_metrics(LYRICS_BUNDLE["config"]))


@app.route("/predict_lyrics", methods=["POST"])
def predict_lyrics_api():
    if LYRICS_BUNDLE["model"] is None:
        return jsonify({"message": "歌词模型尚未训练或模型文件不存在"}), 404

    data = request.get_json(silent=True) or {}
    lyrics_text = (data.get("lyrics_text") or "").strip()
    if not lyrics_text:
        return jsonify({"message": "请提供 lyrics_text"}), 400

    predicted_class, probabilities, diagnostics = predict_lyrics(
        LYRICS_BUNDLE["model"],
        lyrics_text,
        pretrained_model_name=LYRICS_BUNDLE["config"].get("pretrained_model_name", "bert-base-uncased"),
        max_length=LYRICS_BUNDLE["config"].get("max_length", 256),
    )

    predicted_label = LYRICS_BUNDLE["label_mapper"].get_label(predicted_class)
    return jsonify({
        "task_type": "lyrics",
        "genre": predicted_label,
        "probabilities": probabilities_to_response(probabilities),
        "diagnostics": diagnostics,
        "model_config": LYRICS_BUNDLE["config"],
    })


@app.route("/upload_music", methods=["POST"])
def upload_music():
    if AUDIO_BUNDLE["model"] is None:
        return jsonify({"message": "音频模型尚未训练或模型文件不存在"}), 404

    data = request.get_json(silent=True) or {}
    song_name = (data.get("songName") or "未命名歌曲").strip()
    singer_name = (data.get("singerName") or "未知歌手").strip()
    music_base64 = data.get("musicFile")
    user_id = data.get("userId")

    if not music_base64:
        return jsonify({"message": "未收到音乐文件"}), 400
    if not user_id:
        return jsonify({"message": "未收到用户ID"}), 400

    try:
        music_binary = decode_base64_audio(music_base64)
        music_file = io.BytesIO(music_binary)

        predicted_class, probabilities = preprocess_and_predict_file(
            AUDIO_BUNDLE["model"],
            music_file,
            target_sr=AUDIO_BUNDLE["config"]["target_sr"],
            n_mfcc=AUDIO_BUNDLE["config"]["n_mfcc"],
            n_mels=AUDIO_BUNDLE["config"]["n_mels"],
            max_length=AUDIO_BUNDLE["config"]["max_length"],
            feature_type=AUDIO_BUNDLE["config"].get("feature_type", "mfcc"),
            model_type=AUDIO_BUNDLE["config"].get("model_type", "single"),
            standardize=AUDIO_BUNDLE["config"].get("standardize", False),
        )

        if predicted_class is not None and probabilities is not None:
            predicted_label = AUDIO_BUNDLE["label_mapper"].get_label(predicted_class)
            probabilities_response = probabilities_to_response(probabilities)
        else:
            predicted_label = "未知"
            probabilities = [0.0] * len(GENRE_KEYS)
            probabilities_response = probabilities_to_response(probabilities)

        new_music = Music(
            song_name=song_name,
            singer_name=singer_name,
            music_file=music_base64.split("base64,", 1)[1] if "base64," in music_base64 else music_base64,
            face_file=generate_random_image(),
            genre=predicted_label,
            genre_blues=float(probabilities[0]),
            genre_classical=float(probabilities[1]),
            genre_country=float(probabilities[2]),
            genre_disco=float(probabilities[3]),
            genre_hiphop=float(probabilities[4]),
            genre_jazz=float(probabilities[5]),
            genre_metal=float(probabilities[6]),
            genre_pop=float(probabilities[7]),
            genre_reggae=float(probabilities[8]),
            genre_rock=float(probabilities[9]),
            user_id=user_id,
        )
        db.session.add(new_music)
        db.session.commit()

        return jsonify({
            "message": "音乐上传成功",
            "model_type": AUDIO_BUNDLE["config"].get("model_type", "single"),
            "feature_type": AUDIO_BUNDLE["config"].get("feature_type", "mfcc"),
            "genre": predicted_label,
            "probabilities": probabilities_response,
        })
    except Exception as e:
        return jsonify({"message": f"上传失败: {str(e)}"}), 500


@app.route("/predict_multimodal", methods=["POST"])
def predict_multimodal_api():
    data = request.get_json(silent=True) or {}
    music_base64 = data.get("musicFile")
    lyrics_text = (data.get("lyrics_text") or "").strip()

    if not music_base64:
        return jsonify({"message": "未收到音乐文件"}), 400

    try:
        music_binary = decode_base64_audio(music_base64)
        predicted_label, probabilities, fusion_weight = predict_multimodal_from_bytes(music_binary, lyrics_text)
        if predicted_label is None or probabilities is None:
            return jsonify({"message": "多模态模型尚未训练或模型文件不存在"}), 404
        return jsonify({
            "task_type": "multimodal",
            "genre": predicted_label,
            "probabilities": probabilities_to_response(probabilities),
            "fusion_weight": fusion_weight,
        })
    except Exception as e:
        return jsonify({"message": f"多模态预测失败: {str(e)}"}), 500


@app.route("/search_music", methods=["GET"])
def search_music():
    query = request.args.get("query", "").strip()
    if query:
        musics = Music.query.filter(
            or_(
                Music.song_name.ilike(f"%{query}%"),
                Music.genre.ilike(f"%{query}%"),
                Music.singer_name.ilike(f"%{query}%"),
            )
        ).limit(24).all()
    else:
        musics = Music.query.order_by(func.random()).limit(24).all()

    return jsonify([_music_to_dict(music) for music in musics])


@app.route("/music_search_api", methods=["GET"])
def music_search_api():
    keyword = request.args.get("keyword", "").strip()
    if not keyword:
        return jsonify({"message": "请提供 keyword 参数", "results": []}), 400
    results = netease_search_song(keyword, limit=10)
    return jsonify({"keyword": keyword, "results": results})


@app.route("/music_audio_proxy", methods=["GET"])
def music_audio_proxy():
    song_id = request.args.get("id", "").strip()
    if not song_id:
        return jsonify({"message": "缺少 id 参数"}), 400

    url = f"http://music.163.com/song/media/outer/url?id={song_id}.mp3"
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://music.163.com/"}
    try:
        upstream = requests.get(url, headers=headers, stream=True, timeout=15, allow_redirects=True)
    except Exception as e:
        return jsonify({"message": f"音频代理失败: {str(e)}"}), 502

    if upstream.status_code != 200:
        return jsonify({"message": f"上游返回状态码: {upstream.status_code}"}), 502

    content_type = upstream.headers.get("Content-Type", "audio/mpeg")
    return Response(upstream.iter_content(chunk_size=8192), content_type=content_type)


@app.route("/is_favorited", methods=["GET"])
def is_favorited():
    user_id = request.args.get("user_id")
    music_id = request.args.get("music_id")

    if not user_id or not music_id:
        return jsonify({"message": "缺少 user_id 或 music_id"}), 400

    music = db.session.get(Music, music_id)
    if not music:
        return jsonify({"favorited": False})

    if str(music.user_id) == str(user_id):
        return jsonify({"favorited": True})

    exists = db.session.query(Collection.id).filter_by(user_id=user_id, music_id=music_id).first()
    return jsonify({"favorited": bool(exists)})


@app.route("/toggle_favorite", methods=["POST"])
def toggle_favorite():
    data = request.get_json(silent=True) or {}
    user_id = data.get("user_id")
    music_id = data.get("music_id")

    if not user_id or not music_id:
        return jsonify({"message": "缺少 user_id 或 music_id"}), 400

    music = db.session.get(Music, music_id)
    if not music:
        return jsonify({"message": "音乐不存在"}), 404

    if str(music.user_id) == str(user_id):
        return jsonify({"message": "不能收藏自己上传的音乐"}), 403

    collection = Collection.query.filter_by(user_id=user_id, music_id=music_id).first()
    if collection:
        db.session.delete(collection)
        db.session.commit()
        return jsonify({"message": "已取消收藏", "favorited": False})

    new_collection = Collection(user_id=user_id, music_id=music_id)
    db.session.add(new_collection)
    db.session.commit()
    return jsonify({"message": "已收藏", "favorited": True})


@app.route("/register", methods=["POST"])
def register():
    data = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""

    if not username or not password:
        return jsonify({"message": "用户名和密码不能为空"}), 400

    existing_user = User.query.filter_by(username=username).first()
    if existing_user:
        return jsonify({"message": "用户名已存在"}), 400

    new_user = User(
        username=username,
        password=generate_password_hash(password),
        avatar=generate_random_image(),
    )
    db.session.add(new_user)
    db.session.commit()

    return jsonify({"message": "注册成功"}), 201


@app.route("/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""

    if not username or not password:
        return jsonify({"message": "用户名和密码不能为空"}), 400

    user = User.query.filter_by(username=username).first()
    if not user or not verify_password(user.password, password):
        return jsonify({"message": "用户名或密码错误"}), 400

    return jsonify({
        "message": "登录成功",
        "id": user.id,
        "username": user.username,
        "avatar": user.avatar,
        "model_type": AUDIO_BUNDLE["config"].get("model_type", "single"),
        "feature_type": AUDIO_BUNDLE["config"].get("feature_type", "mfcc"),
    }), 200


@app.route("/my_collection", methods=["GET"])
def my_collection():
    user_id = request.args.get("user_id")
    if not user_id:
        return jsonify([])

    results = db.session.query(Music).join(
        Collection, Music.id == Collection.music_id
    ).filter(Collection.user_id == user_id).all()

    return jsonify([_music_to_dict(music) for music in results])


@app.route("/my_uploads", methods=["GET"])
def my_uploads():
    user_id = request.args.get("user_id")
    if not user_id:
        return jsonify([])

    musics = Music.query.filter_by(user_id=user_id).all()
    return jsonify([_music_to_dict(music) for music in musics])


@app.route("/get_music_audio", methods=["GET"])
def get_music_audio():
    music_id = request.args.get("id")
    music = db.session.get(Music, music_id)
    if not music:
        return jsonify({"message": "音乐不存在"}), 404

    return jsonify({"music_file": music.music_file})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)