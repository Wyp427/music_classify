import os
from pathlib import Path

# 数据集路径
ROOT_DIR = Path(r"D:\music_classify_project\dataset_multy2")

# 判断 mp3 是否异常
def check_mp3(file_path):

    try:
        # 1 文件存在
        if not file_path.exists():
            return True, "文件不存在"

        # 2 文件大小检查
        size = file_path.stat().st_size
        if size < 50 * 1024:   # 小于50KB基本不是正常音频
            return True, f"文件过小 ({size} bytes)"

        # 3 检查文件头内容
        with open(file_path, "rb") as f:
            header = f.read(512).lower()

        # HTML错误页
        if b"<html" in header or b"<!doctype" in header:
            return True, "文件实际是HTML"

        # JSON错误页
        if b"error" in header and b"code" in header:
            return True, "疑似API错误返回"

        # 4 检查MP3帧头
        if not (header.startswith(b"ID3") or b"\xff\xfb" in header):
            return True, "没有检测到MP3帧头"

        return False, "正常"

    except Exception as e:
        return True, f"检测异常: {e}"


def scan_dataset():

    print("开始扫描 dataset_multy 中的 mp3 文件...\n")

    total = 0
    bad_count = 0
    good_count = 0

    bad_files = []

    for mp3_file in ROOT_DIR.rglob("*.mp3"):

        total += 1

        bad, reason = check_mp3(mp3_file)

        if bad:
            bad_count += 1
            print(f"[异常] {mp3_file}")
            print(f"       原因: {reason}\n")
            bad_files.append((str(mp3_file), reason))
        else:
            good_count += 1

    print("\n==============================")
    print("扫描完成")
    print(f"总文件数: {total}")
    print(f"正常文件: {good_count}")
    print(f"异常文件: {bad_count}")
    print("==============================")

    # 保存异常列表
    if bad_files:
        output_file = "bad_mp3_list.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            for path, reason in bad_files:
                f.write(f"{path} | {reason}\n")

        print(f"\n异常文件列表已保存: {output_file}")


if __name__ == "__main__":
    scan_dataset()