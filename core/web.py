import base64
import io
import random

import requests
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func, or_
from werkzeug.security import check_password_hash, generate_password_hash

from label_mapper import GTZANLabelMapper
from model_factory import load_model_and_config
from pre_process import preprocess_and_predict_file

#网站web可视化

model, config, _ = load_model_and_config("best_model_config.json", "best_model.pth")
label_mapper = GTZANLabelMapper()

app = Flask(__name__)
CORS(app)


@app.route('/')
def index():
    return """
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>AI音乐风格分类系统</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
body{font-family:Arial;text-align:center;background:#f5f5f5;}
.container{display:flex;justify-content:space-around;margin-top:30px;}
.card{background:white;padding:20px;border-radius:10px;box-shadow:0 0 10px rgba(0,0,0,0.1);width:30%;}
button{padding:10px;margin-top:10px;cursor:pointer;}
.musicCard{background:white;margin:10px;padding:10px;border-radius:10px;}
</style>
</head>
<body>
<h1>🎵 AI音乐风格分类系统</h1>
<p>当前模型类型：<span id="modelType">loading...</span></p>
<p>当特征类型：<span id="featureType">loading...</span></p>
<div class="container">
<div class="card">
<h2>上传音乐</h2>
<input type="text" id="songName" placeholder="歌曲名"><br>
<input type="text" id="singerName" placeholder="歌手"><br>
<input type="file" id="musicFile"><br>
<button onclick="uploadMusic()">上传并分类</button>
</div>
<div class="card">
<h2>音乐播放器</h2>
<audio id="audioPlayer" controls></audio>
</div>
<div class="card">
<h2>分类结果</h2>
<p id="genreResult">等待预测</p>
<canvas id="probChart"></canvas>
</div>
</div>
<hr>
<h2>搜索音乐</h2>
<input type="text" id="searchInput">
<button onclick="searchMusic()">搜索</button>
<div id="musicList"></div>
<script>
let chart = null;

async function uploadMusic(){
    let file = document.getElementById("musicFile").files[0];
    if(!file){ alert("请上传音乐"); return; }
    let reader = new FileReader();
    reader.onload = async function(){
        let base64 = reader.result;
        document.getElementById("audioPlayer").src = base64;
        let data = {
            songName: document.getElementById("songName").value,
            singerName: document.getElementById("singerName").value,
            musicFile: base64,
            userId: 1
        };
        let res = await fetch("/upload_music", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(data)
        });
        let result = await res.json();
        document.getElementById("modelType").innerText = result.model_type || "unknown";
        document.getElementById("featureType").innerText = result.feature_type || "unknown";
        document.getElementById("genreResult").innerText = result.genre ? "该音乐风格为：" + result.genre : (result.message || "预测失败");
        if(result.probabilities){ showProb(result.probabilities); }
    };
    reader.readAsDataURL(file);
}

async function searchMusic(){
    let query = document.getElementById("searchInput").value;
    let res = await fetch(`/search_music?query=${query}`);
    let data = await res.json();
    let list = document.getElementById("musicList");
    list.innerHTML = "";
    data.forEach(m => {
        let div = document.createElement("div");
        div.className = "musicCard";
        div.innerHTML = `
            <h3>${m.song_name}</h3>
            <p>歌手: ${m.singer_name}</p>
            <p>风格: ${m.genre}</p>
            <button onclick="playMusic(${m.id})">播放</button>
            <button onclick="favorite(${m.id})">收藏</button>
            <button onclick='showProb(${JSON.stringify(m.genreProbabilities)})'>概率</button>`;
        list.appendChild(div);
    });
}

async function playMusic(id){
    let res = await fetch(`/get_music_audio?id=${id}`);
    let data = await res.json();
    document.getElementById("audioPlayer").src = "data:audio/mp3;base64," + data.music_file;
}

async function favorite(id){
    await fetch("/toggle_favorite", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({user_id: 1, music_id: id})
    });
    alert("收藏成功");
}

function showProb(prob){
    let labels = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"];
    let values = labels.map(label => {
        if(prob[`genre_${label}`] !== undefined) return Number(prob[`genre_${label}`]);
        if(prob[label] !== undefined) return Number(prob[label]);
        return 0;
    });
    let ctx = document.getElementById("probChart").getContext("2d");
    if(chart){ chart.destroy(); }
    chart = new Chart(ctx, {
        type: "bar",
        data: {
            labels: labels,
            datasets: [{ label: "概率", data: values, backgroundColor: "rgba(54,162,235,0.6)" }]
        },
        options: { scales: { y: { beginAtZero: true } } }
    });
}
</script>
</body>
</html>
"""


app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:Ww778899654321,./@127.0.0.1:3306/music_classify'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

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
    genre_blues = db.Column(db.String(255), nullable=False)
    genre_classical = db.Column(db.String(255), nullable=False)
    genre_country = db.Column(db.String(255), nullable=False)
    genre_disco = db.Column(db.String(255), nullable=False)
    genre_hiphop = db.Column(db.String(255), nullable=False)
    genre_jazz = db.Column(db.String(255), nullable=False)
    genre_metal = db.Column(db.String(255), nullable=False)
    genre_pop = db.Column(db.String(255), nullable=False)
    genre_reggae = db.Column(db.String(255), nullable=False)
    genre_rock = db.Column(db.String(255), nullable=False)


class Collection(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    music_id = db.Column(db.Integer, nullable=False)
    __table_args__ = (db.UniqueConstraint('user_id', 'music_id', name='unique_user_music'),)


with app.app_context():
    db.create_all()


@app.route('/is_favorited', methods=['GET'])
def is_favorited():
    user_id = request.args.get('user_id')
    music_id = request.args.get('music_id')
    if not user_id or not music_id:
        return jsonify({'message': '缺少 user_id 或 music_id'}), 400
    music = Music.query.get(music_id)
    if not music:
        return jsonify({'favorited': False})
    if str(music.user_id) == str(user_id):
        return jsonify({'favorited': True})
    exists = db.session.query(Collection.id).filter_by(user_id=user_id, music_id=music_id).first()
    return jsonify({'favorited': bool(exists)})


@app.route('/toggle_favorite', methods=['POST'])
def toggle_favorite():
    data = request.get_json()
    user_id = data.get('user_id')
    music_id = data.get('music_id')
    if not user_id or not music_id:
        return jsonify({'message': '缺少 user_id 或 music_id'}), 400
    music = Music.query.get(music_id)
    if not music:
        return jsonify({'message': '音乐不存在'}), 404
    if str(music.user_id) == str(user_id):
        return jsonify({'message': '不能收藏自己上传的音乐'}), 403
    collection = Collection.query.filter_by(user_id=user_id, music_id=music_id).first()
    if collection:
        db.session.delete(collection)
        db.session.commit()
        return jsonify({'message': '已取消收藏', 'favorited': False})
    new_collection = Collection(user_id=user_id, music_id=music_id)
    db.session.add(new_collection)
    db.session.commit()
    return jsonify({'message': '已收藏', 'favorited': True})


def generate_random_image():
    index = random.randint(1, 1000)
    image_url = f'https://picsum.photos/200/200?random={index}'
    response = requests.get(image_url)
    if response.status_code == 200:
        return base64.b64encode(response.content).decode('utf-8')
    return ''


@app.route('/upload_music', methods=['POST'])
def upload_music():
    data = request.get_json()
    song_name = data.get('songName')
    singer_name = data.get('singerName')
    music_base64 = data.get('musicFile')
    user_id = data.get('userId')
    if not music_base64:
        return jsonify({'message': '未收到音乐文件'}), 400
    if not user_id:
        return jsonify({'message': '未收到用户ID'}), 400
    if "base64," in music_base64:
        music_base64 = music_base64.split("base64,")[1]

    try:
        music_binary = base64.b64decode(music_base64)
        music_file = io.BytesIO(music_binary)
        predicted_class, probabilities = preprocess_and_predict_file(
            model,
            music_file,
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
            probabilities_response = {
                'blues': float(probabilities[0]), 'classical': float(probabilities[1]), 'country': float(probabilities[2]),
                'disco': float(probabilities[3]), 'hiphop': float(probabilities[4]), 'jazz': float(probabilities[5]),
                'metal': float(probabilities[6]), 'pop': float(probabilities[7]), 'reggae': float(probabilities[8]),
                'rock': float(probabilities[9]),
            }
        else:
            predicted_label = '未知'
            probabilities = [0.0] * 10
            probabilities_response = {label: 0.0 for label in ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']}

        cover_base64 = generate_random_image()
        new_music = Music(
            song_name=song_name,
            singer_name=singer_name,
            music_file=music_base64,
            face_file=cover_base64,
            genre=predicted_label,
            genre_blues=str(probabilities[0]),
            genre_classical=str(probabilities[1]),
            genre_country=str(probabilities[2]),
            genre_disco=str(probabilities[3]),
            genre_hiphop=str(probabilities[4]),
            genre_jazz=str(probabilities[5]),
            genre_metal=str(probabilities[6]),
            genre_pop=str(probabilities[7]),
            genre_reggae=str(probabilities[8]),
            genre_rock=str(probabilities[9]),
            user_id=user_id,
        )
        db.session.add(new_music)
        db.session.commit()
        return jsonify({
            'message': '音乐上传成功',
            'model_type': config.get('model_type', 'single'),
            'feature_type': config.get('feature_type', 'mfcc'),
            'genre': predicted_label,
            'probabilities': probabilities_response,
        })
    except Exception as e:
        return jsonify({'message': f'上传失败: {str(e)}'}), 500


@app.route('/search_music', methods=['GET'])
def search_music():
    query = request.args.get('query', '').strip()
    if query:
        musics = Music.query.filter(or_(Music.song_name.ilike(f'%{query}%'), Music.genre.ilike(f'%{query}%'))).limit(24).all()
    else:
        musics = Music.query.order_by(func.random()).limit(24).all()
    results = [{
        'id': music.id,
        'song_name': music.song_name,
        'singer_name': music.singer_name,
        'face_file': music.face_file,
        'genre': music.genre,
        'genreProbabilities': {
            'genre_blues': music.genre_blues,
            'genre_classical': music.genre_classical,
            'genre_country': music.genre_country,
            'genre_disco': music.genre_disco,
            'genre_hiphop': music.genre_hiphop,
            'genre_jazz': music.genre_jazz,
            'genre_metal': music.genre_metal,
            'genre_pop': music.genre_pop,
            'genre_reggae': music.genre_reggae,
            'genre_rock': music.genre_rock,
        }
    } for music in musics]
    return jsonify(results)


@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    if not username or not password:
        return jsonify({'message': '用户名和密码不能为空'}), 400
    existing_user = User.query.filter_by(username=username).first()
    if existing_user:
        return jsonify({'message': '用户名已存在'}), 400
    new_user = User(username=username, password=password)
    new_user.avatar = generate_random_image()
    db.session.add(new_user)
    db.session.commit()
    return jsonify({'message': '注册成功'}), 201


@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    if not username or not password:
        return jsonify({'message': '用户名和密码不能为空'}), 400
    user = User.query.filter_by(username=username).first()
    if not user or user.password != password:
        return jsonify({'message': '用户名或密码错误'}), 400
    return jsonify({
        'message': '登录成功',
        'id': user.id,
        'username': user.username,
        'avatar': user.avatar,
        'model_type': config.get('model_type', 'single'),
        'feature_type': config.get('feature_type', 'mfcc'),
    }), 200


@app.route('/my_collection')
def my_collection():
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify([])
    results = db.session.query(Music).join(Collection, Music.id == Collection.music_id).filter(Collection.user_id == user_id).all()
    return jsonify([{
        'id': music.id,
        'name': music.song_name,
        'singer': music.singer_name,
        'cover': music.face_file,
        'genre': music.genre,
        'genreProbabilities': {
            'genre_blues': music.genre_blues,
            'genre_classical': music.genre_classical,
            'genre_country': music.genre_country,
            'genre_disco': music.genre_disco,
            'genre_hiphop': music.genre_hiphop,
            'genre_jazz': music.genre_jazz,
            'genre_metal': music.genre_metal,
            'genre_pop': music.genre_pop,
            'genre_reggae': music.genre_reggae,
            'genre_rock': music.genre_rock,
        }
    } for music in results])


@app.route('/get_music_audio', methods=['GET'])
def get_music_audio():
    music_id = request.args.get('id')
    music = Music.query.get(music_id)
    if not music:
        return jsonify({"error": "Music not found"}), 404
    try:
        return jsonify({"music_file": music.music_file})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/my_uploads')
def my_uploads():
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify([])
    musics = Music.query.filter_by(user_id=user_id).all()
    return jsonify([{
        'id': music.id,
        'name': music.song_name,
        'singer': music.singer_name,
        'cover': music.face_file,
        'genre': music.genre,
        'genreProbabilities': {
            'genre_blues': music.genre_blues,
            'genre_classical': music.genre_classical,
            'genre_country': music.genre_country,
            'genre_disco': music.genre_disco,
            'genre_hiphop': music.genre_hiphop,
            'genre_jazz': music.genre_jazz,
            'genre_metal': music.genre_metal,
            'genre_pop': music.genre_pop,
            'genre_reggae': music.genre_reggae,
            'genre_rock': music.genre_rock,
        }
    } for music in musics])


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)