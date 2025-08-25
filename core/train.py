import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.optim.lr_scheduler import StepLR
from cnn import AudioCNN  # 假设 AudioCNN 已经定义在 cnn.py
import numpy as np
import os
from tqdm import tqdm
import json
from data_process import audio_to_cnn_data  # 导入数据处理函数

# 超参数配置
hyperparameters = {
    "folder_path": './datasets/music',  # 请替换为实际的音频文件夹路径
    "batch_size": 32,
    "learning_rate": 4e-4,
    "num_epochs": 100,
    "train_ratio": 0.8,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "weight_decay": 1e-4,  # 优化器的权重衰减
    "step_size": 20,  # 学习率调度器的步长
    "gamma": 0.678,  # 学习率调度器的衰减因子
    "optimizer": "Adam",  # 可选值: "Adam", "SGD", "Adagrad"
    "momentum": 0.937,  # SGD 优化器的动量
    "nesterov": True,  # SGD 优化器是否使用 Nesterov 动量
    "eps": 1e-8,  # Adagrad 优化器的小常数，避免除零
    "lr_decay": 0.0,  # Adagrad 优化器的学习率衰减
    "random_seed": 42  # 添加随机种子
}

# 设置随机种子
torch.manual_seed(hyperparameters["random_seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(hyperparameters["random_seed"])
np.random.seed(hyperparameters["random_seed"])

# 获取设备
device = torch.device(hyperparameters["device"])
print(f"使用设备: {device}")

# 数据缓存文件路径
cache_file = "processed_data_cache.npz"

# 加载数据
print("加载数据...")

# 检查缓存文件是否存在
if os.path.exists(cache_file):
    print(f"加载缓存数据: {cache_file}...")
    cache = np.load(cache_file)
    data = cache["data"]
    encoded_labels = cache["encoded_labels"]
else:
    # 如果缓存文件不存在，重新处理数据
    folder_path = hyperparameters["folder_path"]
    data, encoded_labels = audio_to_cnn_data(folder_path, cache_file=cache_file)

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

# 初始化模型、损失函数和优化器
model = AudioCNN(num_classes=len(np.unique(encoded_labels))).to(device)
criterion = nn.CrossEntropyLoss()

# 选择优化器
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

# 保存最佳模型
best_val_accuracy = 0
best_model_state = None

# 用于存储训练输出的列表
training_output = []

# 训练模型
num_epochs = hyperparameters["num_epochs"]
for epoch in range(num_epochs):
    model.train()
    train_running_loss = 0.0
    train_correct = 0
    train_total = 0
    with tqdm(train_dataloader, unit="batch") as tepoch:
        for batch_idx, (inputs, targets) in enumerate(tepoch):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_running_loss += loss.item()

            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()

            tepoch.set_postfix(loss=train_running_loss / (batch_idx + 1),
                               train_accuracy=100. * train_correct / train_total)

    # 验证模型
    model.eval()
    val_running_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, targets in val_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_running_loss += loss.item()

            _, predicted = outputs.max(1)
            val_total += targets.size(0)
            val_correct += predicted.eq(targets).sum().item()

    val_accuracy = 100. * val_correct / val_total
    train_loss = train_running_loss / len(train_dataloader)
    train_accuracy = 100. * train_correct / train_total
    val_loss = val_running_loss / len(val_dataloader)

    print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss}, '
          f'Train Accuracy: {train_accuracy}%, '
          f'Val Loss: {val_loss}, Val Accuracy: {val_accuracy}%')

    # 保存当前轮次的训练输出
    training_output.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy
    })

    # 更新学习率
    scheduler.step()

    # 保存最佳模型
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_model_state = model.state_dict()
        torch.save(best_model_state, 'best_model_test.pth')
        print(f"Best model saved at epoch {epoch + 1} with validation accuracy: {val_accuracy}%")

# 将训练输出保存到 JSON 文件
with open('training_output.json', 'w') as f:
    json.dump(training_output, f, indent=4)
