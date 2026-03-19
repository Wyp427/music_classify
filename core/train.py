import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

from cnn import AudioCNN
from data_process import audio_to_cnn_data
from feature_utils import get_input_channels, save_feature_config

"""
本文件负责模型训练流程的完整实现，并支持多种音频特征的对比实验。

在超参数配置中新增 feature_type、n_mfcc、n_mels、max_length 等参数，
使得仅通过修改配置即可切换不同特征类型，实现高效实验管理。

训练数据通过 audio_to_cnn_data 动态构建，并根据特征类型生成独立缓存文件，
避免不同实验之间的数据污染。

模型构建阶段通过 get_input_channels 自动获取输入通道数，
确保网络结构与特征维度一致。

新增 compute_macro_recall 函数，用于计算宏平均召回率，
并在验证阶段统计整体 Recall，以补充仅使用 Accuracy 的不足。

训练日志中记录 feature_type、验证准确率与召回率，
便于后续实验对比与分析。

当模型性能提升时，除保存权重外，还保存 best_model_config.json，
记录完整特征配置（特征类型、采样率、维度参数等），
用于后续测试和部署阶段自动加载。

此外，为每种特征实验单独保存指标文件 experiment_metrics_<feature_type>.json，
便于快速构建实验对比结果表。
"""


# 超参数配置
hyperparameters = {
    "folder_path": "D:/music_classify_project/dataset_multy2_processed/audio",
    "feature_type": "mfcc_mel",  # 可选: mfcc / mel / mfcc_mel
    "n_mfcc": 13,
    "n_mels": 128,
    "max_length": 1000,
    "batch_size": 16,
    "learning_rate": 4e-4,
    "num_epochs": 50,
    "train_ratio": 0.8,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "weight_decay": 1e-4,
    "step_size": 20,
    "gamma": 0.678,
    "optimizer": "Adam",
    "momentum": 0.937,
    "nesterov": True,
    "eps": 1e-8,
    "lr_decay": 0.0,
    "random_seed": 42,
}

# 设置随机种子
torch.manual_seed(hyperparameters["random_seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(hyperparameters["random_seed"])
np.random.seed(hyperparameters["random_seed"])

# 获取设备
device = torch.device(hyperparameters["device"])
print(f"使用设备: {device}")

feature_type = hyperparameters["feature_type"]
cache_file = f"processed_data_cache_{feature_type}.npz"
print(f"当前特征类型: {feature_type}")
print("加载数据...")

if os.path.exists(cache_file):
    print(f"加载缓存数据: {cache_file}...")
    cache = np.load(cache_file)
    data = cache["data"]
    encoded_labels = cache["encoded_labels"]
else:
    folder_path = hyperparameters["folder_path"]
    data, encoded_labels = audio_to_cnn_data(
        folder_path,
        target_sr=22050,
        n_mfcc=hyperparameters["n_mfcc"],
        n_mels=hyperparameters["n_mels"],
        max_length=hyperparameters["max_length"],
        feature_type=feature_type,
        cache_file=cache_file,
    )

# 转换为 Tensor
data_tensor = torch.from_numpy(data).float()
label_tensor = torch.from_numpy(encoded_labels).long()
dataset = TensorDataset(data_tensor, label_tensor)

# 划分训练集和验证集
train_size = int(hyperparameters["train_ratio"] * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=hyperparameters["batch_size"], shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=hyperparameters["batch_size"], shuffle=False)

input_channels = get_input_channels(
    feature_type,
    n_mfcc=hyperparameters["n_mfcc"],
    n_mels=hyperparameters["n_mels"],
)
num_classes = len(np.unique(encoded_labels))
model = AudioCNN(num_classes=num_classes, input_channels=input_channels).to(device)
criterion = nn.CrossEntropyLoss()

if hyperparameters["optimizer"] == "Adam":
    optimizer = optim.Adam(model.parameters(), lr=hyperparameters["learning_rate"], weight_decay=hyperparameters["weight_decay"])
elif hyperparameters["optimizer"] == "SGD":
    optimizer = optim.SGD(model.parameters(), lr=hyperparameters["learning_rate"], weight_decay=hyperparameters["weight_decay"],
                          momentum=hyperparameters["momentum"], nesterov=hyperparameters["nesterov"])
elif hyperparameters["optimizer"] == "Adagrad":
    optimizer = optim.Adagrad(model.parameters(), lr=hyperparameters["learning_rate"], weight_decay=hyperparameters["weight_decay"],
                              eps=hyperparameters["eps"], lr_decay=hyperparameters["lr_decay"])
else:
    raise ValueError(f"Unsupported optimizer: {hyperparameters['optimizer']}")

scheduler = StepLR(optimizer, step_size=hyperparameters["step_size"], gamma=hyperparameters["gamma"])



# =========================
# 记录变量
# =========================
best_val_accuracy = 0
best_val_recall = 0

training_output = []

# 👉 新增（平均指标用）
val_accuracy_list = []
val_recall_list = []

# =========================
# Recall计算
# =========================
def compute_macro_recall(targets, predictions, num_classes):
    recalls = []
    targets = np.array(targets)
    predictions = np.array(predictions)

    for class_idx in range(num_classes):
        mask = targets == class_idx
        total = mask.sum()
        if total == 0:
            continue
        tp = ((predictions == class_idx) & mask).sum()
        recalls.append(tp / total)

    return float(np.mean(recalls)) if recalls else 0.0


# =========================
# 训练循环
# =========================
for epoch in range(hyperparameters["num_epochs"]):

    # ===== Train =====
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0

    with tqdm(train_dataloader) as pbar:
        pbar.set_description(f"Epoch {epoch+1}/{hyperparameters['num_epochs']}")

        for i, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, pred = outputs.max(1)

            train_total += targets.size(0)
            train_correct += pred.eq(targets).sum().item()

            pbar.set_postfix(
                loss=train_loss/(i+1),
                train_accuracy=100.*train_correct/train_total
            )

    # ===== Validation =====
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0

    val_targets_all = []
    val_preds_all = []

    with torch.no_grad():
        for inputs, targets in val_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()

            _, pred = outputs.max(1)
            val_total += targets.size(0)
            val_correct += pred.eq(targets).sum().item()

            val_targets_all.extend(targets.cpu().numpy())
            val_preds_all.extend(pred.cpu().numpy())

    val_accuracy = 100 * val_correct / val_total
    val_recall = 100 * compute_macro_recall(val_targets_all, val_preds_all, num_classes)

    # 👉 新增记录
    val_accuracy_list.append(val_accuracy)
    val_recall_list.append(val_recall)

    print(f"Train Acc: {100*train_correct/train_total:.2f}% | "
          f"Val Acc: {val_accuracy:.2f}% | Val Recall: {val_recall:.2f}%")

    training_output.append({
        "epoch": epoch+1,
        "feature_type": feature_type,
        "val_accuracy": val_accuracy,
        "val_recall": val_recall
    })

    scheduler.step()

    # ===== 保存最佳模型 =====
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_val_recall = val_recall

        torch.save(model.state_dict(), "best_model.pth")

        save_feature_config("best_model_config.json", {
            "feature_type": feature_type,
            "target_sr": 22050,
            "n_mfcc": hyperparameters["n_mfcc"],
            "n_mels": hyperparameters["n_mels"],
            "max_length": hyperparameters["max_length"],
            "num_classes": num_classes,
            "input_channels": input_channels,
        })

        print(f"✔ Best model saved (Acc={val_accuracy:.2f}%)")

# =========================
# 平均指标
# =========================
avg_val_accuracy = float(np.mean(val_accuracy_list))
avg_val_recall = float(np.mean(val_recall_list))

# =========================
# 保存实验结果
# =========================
save_feature_config(f"experiment_metrics_{feature_type}.json", {
    "feature_type": feature_type,
    "best_val_accuracy": best_val_accuracy,
    "best_val_recall": best_val_recall,
    "avg_val_accuracy": avg_val_accuracy,
    "avg_val_recall": avg_val_recall,
})

# =========================
# 保存训练日志
# =========================
with open("training_output.json", "w", encoding="utf-8") as f:
    json.dump(training_output, f, indent=4, ensure_ascii=False)

print("训练完成 ✅")