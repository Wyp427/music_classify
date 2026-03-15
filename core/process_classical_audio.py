import warnings
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

warnings.filterwarnings("ignore")


class ClassicalAudioProcessor:

    def __init__(
        self,
        input_dir="D:/music_classify_project/dataset_multy2/classical/music",
        output_dir="D:/music_classify_project/dataset_multy2_processed/audio/classical",
        max_samples=100,
    ):

        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)

        self.target_sr = 22050
        self.target_duration = 30
        self.target_length = self.target_sr * self.target_duration

        self.max_samples = max_samples

        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ==========================
    # 音频处理
    # ==========================
    def process_audio(self, input_path, output_path):

        try:

            audio, _ = librosa.load(
                input_path,
                sr=self.target_sr,
                mono=True,
            )

            if len(audio) == 0:
                raise ValueError("音频为空")

            # 长度统一
            if len(audio) < self.target_length:

                repeat = self.target_length // len(audio) + 1
                audio = np.tile(audio, repeat)[: self.target_length]

            else:

                start = (len(audio) - self.target_length) // 2
                audio = audio[start:start + self.target_length]

            sf.write(
                output_path,
                audio,
                self.target_sr,
                subtype="PCM_16",
            )

            return True

        except Exception as e:

            print(f"处理失败 {input_path}: {e}")
            return False

    # ==========================
    # 处理classical
    # ==========================
    def process(self):

        print("开始处理 classical 音频")

        mp3_files = sorted(self.input_dir.glob("*.mp3"))

        count = 0

        for mp3_file in tqdm(mp3_files):

            if count >= self.max_samples:
                break

            sample_id = f"classical_{count:02d}"

            output_path = self.output_dir / f"{sample_id}.wav"

            if self.process_audio(mp3_file, output_path):

                count += 1

        print("\n处理完成")
        print("生成数量:", count)
        print("输出目录:", self.output_dir)


def main():

    processor = ClassicalAudioProcessor()
    processor.process()


if __name__ == "__main__":
    main()