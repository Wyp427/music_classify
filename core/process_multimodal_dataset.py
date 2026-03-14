import os
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
import json
import re
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


class MultimodalDatasetProcessor:
    def __init__(self, input_dir="D:/music_classify_project/dataset_multy",
                 output_dir="./datasets/multimodal"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.target_sr = 22050  # 目标采样率
        self.target_duration = 30  # 目标时长（秒）
        self.target_length = self.target_duration * self.target_sr

        # 创建输出目录结构
        self.audio_dir = self.output_dir / "audio"
        self.lyrics_dir = self.output_dir / "lyrics"
        self.metadata_dir = self.output_dir / "metadata"

        for dir_path in [self.audio_dir, self.lyrics_dir, self.metadata_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

            # 为每个流派创建子目录
        self.genres = ['blues', 'classical', 'country', 'disco', 'hiphop',
                       'jazz', 'metal', 'pop', 'reggae', 'rock']
        for genre in self.genres:
            (self.audio_dir / genre).mkdir(exist_ok=True)
            (self.lyrics_dir / genre).mkdir(exist_ok=True)

    def clean_lrc_lyrics(self, lrc_content):
        """清洗LRC歌词文件，移除时间标签和元数据"""
        lines = lrc_content.split('\n')
        cleaned_lines = []

        for line in lines:
            # 移除时间标签 [mm:ss.xx]
            line = re.sub(r'\[\d{2}:\d{2}\.\d{2,3}\]', '', line)
            # 移除元数据标签 [ar:], [ti:], [al:], [by:], [offset:]
            line = re.sub(r'\[\w+:[^\]]*\]', '', line)
            # 清理空白字符
            line = line.strip()

            if line and not line.startswith('['):
                cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)

    def process_audio_to_standard_format(self, input_path, output_path):
        """将音频处理为标准格式：30秒，22050Hz，16位单声道WAV"""
        try:
            # 加载音频并重采样
            audio, sr = librosa.load(input_path, sr=self.target_sr, mono=True)

            # 处理长度：截取或循环填充到30秒
            if len(audio) < self.target_length:
                # 循环填充
                repeat_times = self.target_length // len(audio) + 1
                audio = np.tile(audio, repeat_times)[:self.target_length]
            else:
                # 截取中间30秒
                start = (len(audio) - self.target_length) // 2
                audio = audio[start:start + self.target_length]

                # 保存为16位WAV格式
            sf.write(output_path, audio, self.target_sr, subtype='PCM_16')
            return True

        except Exception as e:
            print(f"处理音频失败 {input_path}: {e}")
            return False

    def process_dataset(self):
        """处理整个数据集"""
        print("开始处理多模态数据集...")

        song_mapping = {}
        genre_counts = {genre: 0 for genre in self.genres}

        for genre in self.genres:
            print(f"\n处理 {genre} 流派...")

            genre_input_dir = self.input_dir / genre
            lyric_dir = genre_input_dir / "lyric"
            music_dir = genre_input_dir / "music"

            if not lyric_dir.exists() or not music_dir.exists():
                print(f"警告: {genre} 目录结构不完整，跳过")
                continue

                # 获取所有LRC文件
            lrc_files = list(lyric_dir.glob("*.lrc"))

            for lrc_file in tqdm(lrc_files, desc=f"处理{genre}"):
                base_name = lrc_file.stem

                # 查找对应的MP3文件
                mp3_file = music_dir / f"{base_name}.mp3"
                if not mp3_file.exists():
                    print(f"警告: 找不到对应的MP3文件 {mp3_file}")
                    continue

                    # 处理歌词
                try:
                    with open(lrc_file, 'r', encoding='utf-8') as f:
                        lrc_content = f.read()

                    cleaned_lyrics = self.clean_lrc_lyrics(lrc_content)

                    # 保存清洗后的歌词
                    output_lyric_path = self.lyrics_dir / genre / f"{genre}_{genre_counts[genre]:04d}.txt"
                    with open(output_lyric_path, 'w', encoding='utf-8') as f:
                        f.write(cleaned_lyrics)

                except Exception as e:
                    print(f"处理歌词失败 {lrc_file}: {e}")
                    continue

                    # 处理音频
                output_audio_path = self.audio_dir / genre / f"{genre}_{genre_counts[genre]:04d}.wav"
                if not self.process_audio_to_standard_format(mp3_file, output_audio_path):
                    continue

                    # 记录映射关系
                song_id = f"{genre}_{genre_counts[genre]:04d}"
                song_mapping[song_id] = {
                    "original_audio": str(mp3_file),
                    "original_lyric": str(lrc_file),
                    "processed_audio": str(output_audio_path),
                    "processed_lyric": str(output_lyric_path),
                    "genre": genre
                }

                genre_counts[genre] += 1

                # 保存映射文件
        mapping_file = self.metadata_dir / "song_mapping.json"
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(song_mapping, f, indent=2, ensure_ascii=False)

            # 保存数据集信息
        dataset_info = {
            "total_songs": sum(genre_counts.values()),
            "genre_distribution": genre_counts,
            "audio_format": "16-bit mono WAV, 22050Hz, 30 seconds",
            "lyrics_format": "UTF-8 text, cleaned LRC"
        }

        info_file = self.metadata_dir / "dataset_info.json"
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, indent=2, ensure_ascii=False)

        print(f"\n处理完成！")
        print(f"总歌曲数: {dataset_info['total_songs']}")
        print(f"流派分布: {genre_counts}")
        print(f"输出目录: {self.output_dir}")


def main():
    processor = MultimodalDatasetProcessor()
    processor.process_dataset()


if __name__ == "__main__":
    main()