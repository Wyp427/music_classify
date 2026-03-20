import json
from pathlib import Path

import matplotlib.pyplot as plt


TRAINING_OUTPUT_FILE = Path('training_output.json')
OUTPUT_IMAGE_FILE = Path('training_metrics_plot.png')


try:
    with open(TRAINING_OUTPUT_FILE, 'r', encoding='utf-8') as file:
        data = json.load(file)
except FileNotFoundError:
    print(f"错误: 未找到 '{TRAINING_OUTPUT_FILE}' 文件。")
except json.JSONDecodeError:
    print(f"错误: 无法解析 '{TRAINING_OUTPUT_FILE}' 文件的 JSON 内容。")
else:
    if not data:
        print("错误: training_output.json 内容为空，无法生成报表。")
    else:
        # 基础训练指标
        epochs = [d["epoch"] for d in data]
        train_loss = [d["train_loss"] for d in data]
        train_accuracy = [d["train_accuracy"] for d in data]
        val_loss = [d["val_loss"] for d in data]
        val_accuracy = [d["val_accuracy"] for d in data]
        val_recall = [d.get("val_recall", 0.0) for d in data]

        # 新版训练配置说明信息
        first_record = data[0]
        model_type = first_record.get("model_type", "single")
        feature_type = first_record.get("feature_type", "mfcc")
        fusion_type = first_record.get("fusion_type", "concat")
        standardize = first_record.get("standardize", False)

        best_val_accuracy = max(val_accuracy)
        best_val_recall = max(val_recall)
        avg_val_accuracy = sum(val_accuracy) / len(val_accuracy)
        avg_val_recall = sum(val_recall) / len(val_recall)

        # 创建 2x3 子图
        fig, axs = plt.subplots(2, 3, figsize=(16, 9))
        fig.suptitle('Training Report', fontsize=16)

        # 绘制 train_loss
        axs[0, 0].plot(epochs, train_loss, 'r-')
        axs[0, 0].set_title('Train Loss')
        axs[0, 0].set_xlabel('Epoch')
        axs[0, 0].set_ylabel('Loss')

        # 绘制 train_accuracy
        axs[0, 1].plot(epochs, train_accuracy, 'g-')
        axs[0, 1].set_title('Train Accuracy')
        axs[0, 1].set_xlabel('Epoch')
        axs[0, 1].set_ylabel('Accuracy (%)')

        # 绘制 val_loss
        axs[0, 2].plot(epochs, val_loss, 'b-')
        axs[0, 2].set_title('Validation Loss')
        axs[0, 2].set_xlabel('Epoch')
        axs[0, 2].set_ylabel('Loss')

        # 绘制 val_accuracy
        axs[1, 0].plot(epochs, val_accuracy, 'y-')
        axs[1, 0].set_title('Validation Accuracy')
        axs[1, 0].set_xlabel('Epoch')
        axs[1, 0].set_ylabel('Accuracy (%)')

        # 绘制 val_recall
        axs[1, 1].plot(epochs, val_recall, 'm-')
        axs[1, 1].set_title('Validation Recall')
        axs[1, 1].set_xlabel('Epoch')
        axs[1, 1].set_ylabel('Recall (%)')

        # 右下角放实验信息摘要
        axs[1, 2].axis('off')
        summary_text = (
            f"Model Type: {model_type}\n"
            f"Feature Type: {feature_type}\n"
            f"Fusion Type: {fusion_type}\n"
            f"Standardize: {standardize}\n\n"
            f"Best Val Accuracy: {best_val_accuracy:.2f}%\n"
            f"Best Val Recall: {best_val_recall:.2f}%\n"
            f"Avg Val Accuracy: {avg_val_accuracy:.2f}%\n"
            f"Avg Val Recall: {avg_val_recall:.2f}%"
        )
        axs[1, 2].text(
            0.02,
            0.98,
            summary_text,
            transform=axs[1, 2].transAxes,
            va='top',
            ha='left',
            fontsize=11,
            bbox=dict(boxstyle='round', facecolor='whitesmoke', edgecolor='gray')
        )

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        try:
            plt.savefig(OUTPUT_IMAGE_FILE)
            print(f"图片已成功保存为 '{OUTPUT_IMAGE_FILE}'。")
        except Exception as e:
            print(f"错误: 保存图片时出现问题: {e}")