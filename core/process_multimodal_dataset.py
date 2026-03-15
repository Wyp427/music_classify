import json
import re
import warnings
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

warnings.filterwarnings("ignore")


class MultimodalDatasetProcessor:
    def __init__(
        self,
        input_dir="D:/music_classify_project/dataset_multy2",
        output_dir="D:/music_classify_project/dataset_multy2_processed",
        max_samples_per_genre=100,
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)

        # 音频标准
        self.target_sr = 22050
        self.target_duration = 30
        self.target_length = self.target_duration * self.target_sr

        self.max_samples_per_genre = max_samples_per_genre

        # 输出目录
        self.audio_dir = self.output_dir / "audio"
        self.lyrics_dir = self.output_dir / "lyrics"
        self.metadata_dir = self.output_dir / "metadata"

        for d in [self.audio_dir, self.lyrics_dir, self.metadata_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # 10种音乐风格
        self.genres = [
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

        for genre in self.genres:
            (self.audio_dir / genre).mkdir(exist_ok=True)
            (self.lyrics_dir / genre).mkdir(exist_ok=True)

    # ==========================
    # 歌词清洗函数
    # ==========================
    @staticmethod
    def clean_lrc_lyrics(lrc_content):
        """
        清洗歌词：
        1 去除时间标签
        2 去除元数据
        3 去除中文信息
        4 去除标点符号
        5 转换为小写
        6 清理多余空格
        """

        lines = lrc_content.split("\n")
        cleaned_lines = []

        for line in lines:

            # 去时间标签
            line = re.sub(r"\[\d{2}:\d{2}\.\d{2,3}\]", "", line)

            # 去 metadata
            line = re.sub(r"\[\w+:[^\]]*\]", "", line)

            # 去中文信息
            line = re.sub(
                r"(作词|作曲|编曲|制作人|词|曲|演唱|歌手|音乐制作|混音|录音|母带|监制).*",
                "",
                line,
            )

            # 去中文字符
            line = re.sub(r"[\u4e00-\u9fff]+", "", line)

            # 标点符号替换为空格
            line = re.sub(r"[^a-zA-Z\s]", " ", line)

            # 转换为小写
            line = line.lower()

            # 多余空格清理
            line = re.sub(r"\s+", " ", line).strip()

            if line:
                cleaned_lines.append(line)

        return "\n".join(cleaned_lines)

    # ==========================
    # 音频处理
    # ==========================
    def process_audio_to_standard_format(self, input_path, output_path):

        try:

            audio, _ = librosa.load(
                input_path,
                sr=self.target_sr,
                mono=True,
            )

            if len(audio) == 0:
                raise ValueError("音频为空")

            # 长度处理
            if len(audio) < self.target_length:

                repeat = self.target_length // len(audio) + 1
                audio = np.tile(audio, repeat)[: self.target_length]

            else:

                start = (len(audio) - self.target_length) // 2
                audio = audio[start : start + self.target_length]

            sf.write(
                output_path,
                audio,
                self.target_sr,
                subtype="PCM_16",
            )

            return True

        except Exception as e:

            print(f"处理音频失败 {input_path}: {e}")
            return False

    # ==========================
    # 数据集处理
    # ==========================
    def process_dataset(self):

        print("开始处理多模态数据集...")

        song_mapping = {}
        genre_counts = {g: 0 for g in self.genres}

        for genre in self.genres:

            print(f"\n处理 {genre} 流派...")

            lyric_dir = self.input_dir / genre / "lyric"
            music_dir = self.input_dir / genre / "music"

            if not lyric_dir.exists() or not music_dir.exists():
                print(f"警告: {genre} 目录结构不完整")
                continue

            lrc_files = sorted(lyric_dir.glob("*.lrc"))

            for lrc_file in tqdm(lrc_files, desc=f"处理{genre}"):

                if genre_counts[genre] >= self.max_samples_per_genre:
                    break

                base_name = lrc_file.stem
                mp3_file = music_dir / f"{base_name}.mp3"

                if not mp3_file.exists():
                    print(f"找不到音频 {mp3_file}")
                    continue

                sample_id = f"{genre}_{genre_counts[genre]:02d}"

                output_lyric = self.lyrics_dir / genre / f"{sample_id}.txt"
                output_audio = self.audio_dir / genre / f"{sample_id}.wav"

                # 处理歌词
                try:

                    with open(lrc_file, "r", encoding="utf-8") as f:
                        lrc = f.read()

                    cleaned = self.clean_lrc_lyrics(lrc)

                    with open(output_lyric, "w", encoding="utf-8") as f:
                        f.write(cleaned)

                except Exception as e:

                    print(f"歌词处理失败 {lrc_file}: {e}")
                    continue

                # 处理音频
                if not self.process_audio_to_standard_format(
                    mp3_file, output_audio
                ):

                    if output_lyric.exists():
                        output_lyric.unlink()

                    continue

                # 记录映射
                song_mapping[sample_id] = {

                    "genre": genre,

                    "original_audio": str(mp3_file),
                    "original_lyric": str(lrc_file),

                    "processed_audio": str(output_audio),
                    "processed_lyric": str(output_lyric),
                }

                genre_counts[genre] += 1

        # ==========================
        # 保存metadata
        # ==========================

        mapping_file = self.metadata_dir / "song_mapping.json"

        with open(mapping_file, "w", encoding="utf-8") as f:
            json.dump(song_mapping, f, indent=2, ensure_ascii=False)

        dataset_info = {

            "total_songs": sum(genre_counts.values()),

            "genre_distribution": genre_counts,

            "audio_format": "30s 22050Hz mono 16bit wav",

            "naming_format": "{genre}_00 ~ {genre}_99",

            "lyrics_format": "clean lowercase english text",
        }

        info_file = self.metadata_dir / "dataset_info.json"

        with open(info_file, "w", encoding="utf-8") as f:
            json.dump(dataset_info, f, indent=2, ensure_ascii=False)

        print("\n处理完成！")
        print("总歌曲数:", dataset_info["total_songs"])
        print("流派分布:", genre_counts)
        print("输出目录:", self.output_dir)


def main():

    processor = MultimodalDatasetProcessor()
    processor.process_dataset()


if __name__ == "__main__":
    main()