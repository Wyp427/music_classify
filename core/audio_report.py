import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

TRAINING_OUTPUT_FILE = Path('training_output.json')
TEST_PRED_FILE = Path('test_predictions.json')
TEST_RESULT_FILE = Path('test_results.json')


# =========================
# 读取数据
# =========================
with open(TRAINING_OUTPUT_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)

# ⭐ 自动分 fold（关键修复）
folds = []
current_fold = []

prev_epoch = 0
for d in data:
    if d["epoch"] == 1 and prev_epoch != 0:
        folds.append(current_fold)
        current_fold = []
    current_fold.append(d)
    prev_epoch = d["epoch"]

if current_fold:
    folds.append(current_fold)


# =========================
# 创建图
# =========================
fig, axs = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle("Training Report (3-Fold Cross Validation)", fontsize=18)


# =========================
# 绘制函数
# =========================
def plot_metric(ax, key, title):
    for i, fold in enumerate(folds):
        epochs = [d["epoch"] for d in fold]
        values = [d[key] for d in fold]
        ax.plot(epochs, values, label=f"Fold {i+1}")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.legend()


# =========================
# 第一行
# =========================
plot_metric(axs[0, 0], "train_loss", "Train Loss")
plot_metric(axs[0, 1], "train_accuracy", "Train Accuracy")
plot_metric(axs[0, 2], "val_loss", "Validation Loss")
plot_metric(axs[0, 3], "val_accuracy", "Validation Accuracy")

# =========================
# 第二行
# =========================
plot_metric(axs[1, 0], "val_recall", "Validation Recall")
plot_metric(axs[1, 1], "val_f1", "Validation F1")


# =========================
# 混淆矩阵
# =========================
axs[1, 2].set_title("Confusion Matrix")

if TEST_PRED_FILE.exists():
    with open(TEST_PRED_FILE, 'r') as f:
        pred = json.load(f)

    cm = confusion_matrix(pred["targets"], pred["predictions"])

    im = axs[1, 2].imshow(cm)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axs[1, 2].text(j, i, cm[i, j], ha="center", va="center", fontsize=8)

    axs[1, 2].set_xlabel("Predicted")
    axs[1, 2].set_ylabel("True")
else:
    axs[1, 2].text(0.5, 0.5, "No Data", ha='center')


# =========================
# Summary（论文级）
# =========================
axs[1, 3].axis('off')

all_val_acc = [d["val_accuracy"] for d in data]
all_val_f1 = [d["val_f1"] for d in data]
learning_rate = data[0].get("learning_rate", 0)

lr_str = f"{learning_rate:.0e}"   # 根据不同学习率设置对应的命名
OUTPUT_IMAGE_FILE = Path(f'audio_training_metrics_lr_{lr_str}.png')

if TEST_RESULT_FILE.exists():
    with open(TEST_RESULT_FILE, 'r') as f:
        test_data = json.load(f)

    test_accs = test_data["test_accs"]
    test_f1s = test_data["test_f1s"]

    avg_test_acc = np.mean(test_accs)
    avg_test_f1 = np.mean(test_f1s)

    best_test_acc = max(test_accs)
    best_test_f1 = max(test_f1s)
else:
    avg_test_acc = avg_test_f1 = 0
    best_test_acc = best_test_f1 = 0
summary_text = (
    "Model: Dual-Branch CNN\n"
    "Feature: MFCC + Mel\n"
    "Fusion: Gated\n"
    "Standardize: True\n\n"
    f"Learning Rate: {learning_rate:.6f}\n\n"
    f"Best Val Acc: {max(all_val_acc):.2f}%\n"
    f"Best Val F1: {max(all_val_f1):.2f}%\n\n"
    f"Avg Test Acc: {avg_test_acc:.2f}%\n"
    f"Avg Test F1: {avg_test_f1:.2f}%"
)

axs[1, 3].text(
    0.02, 0.98,
    summary_text,
    va='top',
    fontsize=11,
    bbox=dict(boxstyle='round', facecolor='whitesmoke')
)


# =========================
# 保存
# =========================
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(OUTPUT_IMAGE_FILE)

print("训练结果曲线图已生成")