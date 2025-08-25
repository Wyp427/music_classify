from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
import base64
import io
import random
import requests
import torch
from sqlalchemy import func, or_
from cnn import AudioCNN
from pre_process import preprocess_and_predict, preprocess_and_predict_file
from label_mapper import GTZANLabelMapper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AudioCNN()
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.to(device)
model.eval()
label_mapper = GTZANLabelMapper()

app = Flask(__name__)
CORS(app)

app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:Ww778899654321,./@127.0.0.1:3306/music_classify'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(255), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    avatar = db.Column(db.String(255), nullable=True)  # 添加 avatar_url 字段

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

    # 查询音乐信息
    music = Music.query.get(music_id)
    if not music:
        return jsonify({'favorited': False})  # 音乐不存在也认为未收藏

    # 如果是自己上传的音乐，视为已收藏
    if str(music.user_id) == str(user_id):
        return jsonify({'favorited': True})

    # 查询是否已收藏
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
    else:
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
    user_id = data.get('userId')  # 获取传入的 userId

    if not music_base64:
        return jsonify({'message': '未收到音乐文件'}), 400

    if not user_id:
        return jsonify({'message': '未收到用户ID'}), 400

    if "base64," in music_base64:
        music_base64 = music_base64.split("base64,")[1]

    try:
        music_binary = base64.b64decode(music_base64)
        music_file = io.BytesIO(music_binary)
        predicted_class, probabilities = preprocess_and_predict(model, music_file)

        if predicted_class is not None:
            predicted_label = label_mapper.get_label(predicted_class)
        else:
            predicted_label = '未知'

        cover_base64 = generate_random_image()

        # 创建新音乐记录并关联用户ID
        new_music = Music(
            song_name=song_name,
            singer_name=singer_name,
            music_file=music_base64,
            face_file=cover_base64,
            genre=predicted_label,
            genre_blues=probabilities[0],
            genre_classical=probabilities[1],
            genre_country=probabilities[2],
            genre_disco=probabilities[3],
            genre_hiphop=probabilities[4],
            genre_jazz=probabilities[5],
            genre_metal=probabilities[6],
            genre_pop=probabilities[7],
            genre_reggae=probabilities[8],
            genre_rock=probabilities[9],
            user_id=user_id  # 将 user_id 加入到音乐记录中
        )

        db.session.add(new_music)
        db.session.commit()

        return jsonify({'message': '音乐上传成功'})
    except Exception as e:
        return jsonify({'message': f'上传失败: {str(e)}'}), 500


@app.route('/search_music', methods=['GET'])
def search_music():
    query = request.args.get('query', '').strip()

    if query:
        musics = Music.query.filter(
            or_(
                Music.song_name.ilike(f'%{query}%'),
                Music.genre.ilike(f'%{query}%')
            )
        ).limit(24).all()
    else:
        musics = Music.query.order_by(func.random()).limit(24).all()

    results = [{
        'id': music.id,
        'song_name': music.song_name,
        'singer_name': music.singer_name,
        #'music_file': music.music_file,
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

    # 生成一个随机头像
    random_avatar = generate_random_image()
    new_user.avatar = random_avatar  # 将随机头像 URL 存储到用户信息中

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

    if not user:
        return jsonify({'message': '用户名或密码错误'}), 400

    if user.password != password:
        return jsonify({'message': '用户名或密码错误'}), 400

    return jsonify({
        'message': '登录成功',
        'id': user.id,
        'username': user.username,
        'avatar': user.avatar
    }), 200

@app.route('/my_collection')
def my_collection():
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify([])

    results = db.session.query(Music).join(Collection, Music.id == Collection.music_id) \
        .filter(Collection.user_id == user_id).all()

    return jsonify([
        {
            'id': music.id,
            'name': music.song_name,
            'singer': music.singer_name,
            #'musicFile': music.music_file,
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
        }
        for music in results
    ])

@app.route('/get_music_audio', methods=['GET'])
def get_music_audio():
    music_id = request.args.get('id')
    music = Music.query.get(music_id)
    if not music:
        return jsonify({"error": "Music not found"}), 404

    # 假设音乐的音频文件存储在数据库或文件系统中
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
    return jsonify([
        {
            'id': music.id,
            'name': music.song_name,
            'singer': music.singer_name,
            #'musicFile': music.music_file,
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
        }
        for music in musics
    ])


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
