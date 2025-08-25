import matplotlib.pyplot as plt
import json

# 读取JSON文件
try:
    with open('training_output.json', 'r') as file:
        data = json.load(file)
except FileNotFoundError:
    print("错误: 未找到 'training_output.json' 文件。")
except json.JSONDecodeError:
    print("错误: 无法解析 'training_output.json' 文件的JSON内容。")
else:
    # 提取epoch、train_loss、train_accuracy、val_loss、val_accuracy
    epochs = [d["epoch"] for d in data]
    train_loss = [d["train_loss"] for d in data]
    train_accuracy = [d["train_accuracy"] for d in data]
    val_loss = [d["val_loss"] for d in data]
    val_accuracy = [d["val_accuracy"] for d in data]

    # 创建4宫格子图
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    # 绘制train_loss
    axs[0, 0].plot(epochs, train_loss, 'r-')
    axs[0, 0].set_title('Train Loss')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')

    # 绘制train_accuracy
    axs[0, 1].plot(epochs, train_accuracy, 'g-')
    axs[0, 1].set_title('Train Accuracy')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Accuracy (%)')

    # 绘制val_loss
    axs[1, 0].plot(epochs, val_loss, 'b-')
    axs[1, 0].set_title('Validation Loss')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Loss')

    # 绘制val_accuracy
    axs[1, 1].plot(epochs, val_accuracy, 'y-')
    axs[1, 1].set_title('Validation Accuracy')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Accuracy (%)')

    # 调整子图之间的间距
    plt.tight_layout()

    # 保存图片到本地
    try:
        plt.savefig('training_metrics_plot.png')
        print("图片已成功保存为 'training_metrics_plot.png'。")
    except Exception as e:
        print(f"错误: 保存图片时出现问题: {e}")