import base64
import io
import json
import os
import random
from pathlib import Path

import requests
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func, or_
from werkzeug.security import check_password_hash, generate_password_hash

from model_factory import load_model_and_config
from pre_process import predict_lyrics, preprocess_and_predict_file

AUDIO_CONFIG_PATH = Path("best_model_config.json")
AUDIO_MODEL_PATH = Path("best_model.pth")
LYRICS_CONFIG_PATH = Path("lyrics_best_model_config.json")
LYRICS_MODEL_PATH = Path("lyrics_best_model.pth")

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

    return {
        f"genre_{GENRE_KEYS[i]}": float(probabilities[i])
        for i in range(min(len(GENRE_KEYS), len(probabilities)))
    }


def get_training_metrics(config):
    training_path = config.get("training_output_path", "training_output.json")
    file_path = Path(training_path)
    if not file_path.exists():
        return []
    try:
        return json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


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


@app.route("/")
def index():
    return """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>AI音乐风格分类系统</title>

<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

<!-- ⭐ Three.js -->
<script src="https://cdn.jsdelivr.net/npm/three@0.128/build/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128/examples/js/loaders/GLTFLoader.js"></script>

<style>
body{
    margin:0;
    font-family:"Segoe UI";
    background:linear-gradient(135deg,#1e1e2f,#2a3a4f);
    color:white;
}

h1{text-align:center;}

.container{
    display:grid;
    grid-template-columns:1fr 1fr 1fr;
    gap:20px;
    padding:20px;
}

.card{
    background:rgba(255,255,255,0.08);
    border-radius:15px;
    padding:20px;
    backdrop-filter:blur(10px);
    box-shadow:0 0 15px rgba(0,0,0,0.3);
}

button{
    background:#00c3ff;
    border:none;
    padding:10px 15px;
    border-radius:8px;
    color:white;
    cursor:pointer;
}

textarea{
    width:100%;
    height:120px;
    border-radius:8px;
}

#assistantBox{
    position:fixed;
    bottom:20px;
    right:20px;
    width:350px;
    height:450px;
}

#assistantCanvas{
    width:100%;
    height:100%;
}

#assistantBubble{
    position:absolute;
    bottom:260px;
    right:0;
    background:#00c3ff;
    padding:8px 12px;
    border-radius:10px;
    display:none;
}
</style>
</head>

<body>

<h1>🎵 AI音乐 / 歌词风格分类系统</h1>

<div class="container">

<div class="card">
<h2>音乐上传</h2>
<input id="songName" placeholder="歌曲名"><br>
<input id="singerName" placeholder="歌手"><br>
<input type="file" id="musicFile"><br>
<button onclick="uploadMusic()">上传</button>
</div>

<div class="card">
<h2>播放器</h2>
<audio id="audioPlayer" controls></audio>
</div>

<div class="card">
<h2>结果</h2>
<p id="genreResult"></p>
<canvas id="probChart"></canvas>
</div>

</div>

<hr>

<div class="container">

<div class="card">
<h2>歌词分类</h2>

<textarea id="lyricsText"></textarea><br>

<input type="file" id="lyricsFile" accept=".txt"><br>

<button onclick="predictLyrics()">预测</button>

<p id="lyricsResult"></p>
</div>

</div>

<!-- ⭐ 3D人物 -->
<div id="assistantBox">
    <canvas id="assistantCanvas"></canvas>
    <div id="assistantBubble">你好 🤖</div>
</div>

<script>

let chart=null
let uploadedLyricsText=""

const labels=["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]

// ===== 歌词上传 =====
document.getElementById("lyricsFile").addEventListener("change",function(){

let file=this.files[0]

let reader=new FileReader()

reader.onload=function(e){
uploadedLyricsText=e.target.result
document.getElementById("lyricsText").value=uploadedLyricsText
}

reader.readAsText(file)
})

// ===== 3D人物 =====
const scene = new THREE.Scene()

const camera = new THREE.PerspectiveCamera(75, 1, 0.1, 1000)

const renderer = new THREE.WebGLRenderer({
canvas: document.getElementById("assistantCanvas"),
alpha: true
})

renderer.setSize(220,260)

const light = new THREE.HemisphereLight(0xffffff, 0x444444)
scene.add(light)
camera.position.set(0, 1.2, 6)      //  ④ 改相机

let mixer

//  加载自己的模型 
const loader = new THREE.GLTFLoader()

loader.load(
    "/static/free_cure_girl.glb",     //  ① 改路径

    function (gltf) {

        const model = gltf.scene

        //  根据模型调整
        model.scale.set(5.5,5.5,5.5)      //  ② 改大小
        model.position.x = 0
        model.position.y = -0.3     //  ③ 改位置
        scene.add(model)

        // 动画
        if (gltf.animations.length > 0) {
            mixer = new THREE.AnimationMixer(model)
            const action = mixer.clipAction(gltf.animations[0])
            action.play()
        }
    },

    undefined,

    function (error) {
        console.error("模型加载失败:", error)
    }
)

const clock = new THREE.Clock()

function animate() {
    requestAnimationFrame(animate)

    if (mixer) {
        mixer.update(clock.getDelta())
    }

    renderer.render(scene, camera)
}

animate()

// 点击交互
const bubble=document.getElementById("assistantBubble")

document.getElementById("assistantCanvas").onclick=()=>{
bubble.style.display="block"
bubble.innerText="🎵 欢迎使用AI音乐系统"
setTimeout(()=>bubble.style.display="none",2000)
}

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