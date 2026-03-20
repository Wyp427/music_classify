import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
import json
from data_process import audio_to_cnn_data
from feature_utils import get_input_channels, save_feature_config
from model_factory import build_model_from_config


#主训练代码
# =========================
# 超参数配置
# =========================
hyperparameters = {
    "folder_path": "D:/music_classify_project/dataset_multy2_processed/audio",
    "model_type": "dual_branch",  # 可选: single / dual_branch
    "feature_type": "mfcc_mel",   # single模式可选: mfcc / mel / mfcc_mel
    "fusion_type": "gated",      # dual_branch模式可选: concat / gated
    "standardize": True,          # 标准化可选： False  / True
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


# =========================
# Recall
# =========================
def compute_macro_recall(targets, predictions, num_classes):
    recalls = []
    targets = np.array(targets)
    predictions = np.array(predictions)
    for class_idx in range(num_classes):
        class_mask = targets == class_idx
        class_total = class_mask.sum()
        if class_total == 0:
            continue
        tp = ((predictions == class_idx) & class_mask).sum()
        recalls.append(tp / class_total)
    return float(np.mean(recalls)) if recalls else 0.0


# =========================
# 优化器
# =========================
def create_optimizer(model):
    if hyperparameters["optimizer"] == "Adam":
        return optim.Adam(model.parameters(),
                          lr=hyperparameters["learning_rate"],
                          weight_decay=hyperparameters["weight_decay"])
    elif hyperparameters["optimizer"] == "SGD":
        return optim.SGD(model.parameters(),
                         lr=hyperparameters["learning_rate"],
                         weight_decay=hyperparameters["weight_decay"],
                         momentum=hyperparameters["momentum"],
                         nesterov=hyperparameters["nesterov"])
    elif hyperparameters["optimizer"] == "Adagrad":
        return optim.Adagrad(model.parameters(),
                             lr=hyperparameters["learning_rate"],
                             weight_decay=hyperparameters["weight_decay"],
                             eps=hyperparameters["eps"],
                             lr_decay=hyperparameters["lr_decay"])
    else:
        raise ValueError("Unsupported optimizer")


# =========================
# 随机种子
# =========================
torch.manual_seed(hyperparameters["random_seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(hyperparameters["random_seed"])
np.random.seed(hyperparameters["random_seed"])

device = torch.device(hyperparameters["device"])
print(f"使用设备: {device}")


# =========================
# 实验命名
# =========================
model_type = hyperparameters["model_type"]
feature_type = hyperparameters["feature_type"]

if model_type == "single":
    experiment_name = feature_type
else:
    std_flag = "std" if hyperparameters["standardize"] else "nostd"
    experiment_name = f"dual_{hyperparameters['fusion_type']}_{std_flag}"

cache_file = f"processed_data_cache_{experiment_name}.npz"

print(f"当前模型类型: {model_type}")
print(f"当前实验名称: {experiment_name}")
print("加载数据...")


# =========================
# 数据加载
# =========================
if model_type == "dual_branch":
    mfcc_data, mel_data, encoded_labels = audio_to_cnn_data(
        hyperparameters["folder_path"],
        target_sr=22050,
        n_mfcc=hyperparameters["n_mfcc"],
        n_mels=hyperparameters["n_mels"],
        max_length=hyperparameters["max_length"],
        model_type="dual_branch",
        standardize=hyperparameters["standardize"],
        cache_file=cache_file,
    )

    mfcc_tensor = torch.from_numpy(mfcc_data).float()
    mel_tensor = torch.from_numpy(mel_data).float()
    label_tensor = torch.from_numpy(encoded_labels).long()
    dataset = TensorDataset(mfcc_tensor, mel_tensor, label_tensor)

else:
    data, encoded_labels = audio_to_cnn_data(
        hyperparameters["folder_path"],
        target_sr=22050,
        n_mfcc=hyperparameters["n_mfcc"],
        n_mels=hyperparameters["n_mels"],
        max_length=hyperparameters["max_length"],
        feature_type=feature_type,
        model_type="single",
        standardize=hyperparameters["standardize"],
        cache_file=cache_file,
    )

    data_tensor = torch.from_numpy(data).float()
    label_tensor = torch.from_numpy(encoded_labels).long()
    dataset = TensorDataset(data_tensor, label_tensor)


# =========================
# 划分数据
# =========================
train_size = int(hyperparameters["train_ratio"] * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset,
                              batch_size=hyperparameters["batch_size"],
                              shuffle=True)

val_dataloader = DataLoader(val_dataset,
                            batch_size=hyperparameters["batch_size"],
                            shuffle=False)


# =========================
# 模型
# =========================
num_classes = len(np.unique(encoded_labels))

config_to_save = {
    "model_type": model_type,
    "feature_type": feature_type,
    "fusion_type": hyperparameters["fusion_type"],
    "standardize": hyperparameters["standardize"],
    "target_sr": 22050,
    "n_mfcc": hyperparameters["n_mfcc"],
    "n_mels": hyperparameters["n_mels"],
    "max_length": hyperparameters["max_length"],
    "num_classes": num_classes,
}

if model_type == "single":
    config_to_save["input_channels"] = get_input_channels(
        feature_type,
        n_mfcc=hyperparameters["n_mfcc"],
        n_mels=hyperparameters["n_mels"],
    )

model = build_model_from_config(config_to_save).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = create_optimizer(model)
scheduler = StepLR(optimizer,
                   step_size=hyperparameters["step_size"],
                   gamma=hyperparameters["gamma"])


# =========================
# 训练
# =========================
best_val_accuracy = 0
training_output = []

for epoch in range(hyperparameters["num_epochs"]):

    model.train()
    train_running_loss = 0.0
    train_correct = 0
    train_total = 0

    with tqdm(train_dataloader, unit="batch") as tepoch:
        tepoch.set_description(f"Epoch {epoch + 1}/{hyperparameters['num_epochs']}")

        for batch_idx, batch in enumerate(tepoch):

            if model_type == "dual_branch":
                mfcc_inputs, mel_inputs, targets = batch
                mfcc_inputs = mfcc_inputs.to(device)
                mel_inputs = mel_inputs.to(device)
                targets = targets.to(device)
                outputs = model(mfcc_inputs, mel_inputs)
            else:
                inputs, targets = batch
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)

            optimizer.zero_grad()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_running_loss += loss.item()

            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()

            tepoch.set_postfix(
                loss=train_running_loss / (batch_idx + 1),
                train_accuracy=100. * train_correct / train_total
            )

    # =========================
    # 验证
    # =========================
    model.eval()
    val_running_loss = 0.0
    val_correct = 0
    val_total = 0

    val_targets_all = []
    val_predictions_all = []

    with torch.no_grad():
        for batch in val_dataloader:

            if model_type == "dual_branch":
                mfcc_inputs, mel_inputs, targets = batch
                mfcc_inputs = mfcc_inputs.to(device)
                mel_inputs = mel_inputs.to(device)
                targets = targets.to(device)
                outputs = model(mfcc_inputs, mel_inputs)
            else:
                inputs, targets = batch
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)

            loss = criterion(outputs, targets)
            val_running_loss += loss.item()

            _, predicted = outputs.max(1)
            val_total += targets.size(0)
            val_correct += predicted.eq(targets).sum().item()

            val_targets_all.extend(targets.cpu().numpy())
            val_predictions_all.extend(predicted.cpu().numpy())

    # =========================
    # 指标
    # =========================
    val_accuracy = 100. * val_correct / val_total
    val_recall = 100. * compute_macro_recall(val_targets_all, val_predictions_all, num_classes)

    train_loss = train_running_loss / len(train_dataloader)
    train_accuracy = 100. * train_correct / train_total
    val_loss = val_running_loss / len(val_dataloader)

    print(f'Epoch {epoch + 1}/{hyperparameters["num_epochs"]}, '
          f'Train Loss: {train_loss}, Train Accuracy: {train_accuracy}%, '
          f'Val Loss: {val_loss}, Val Accuracy: {val_accuracy}%, Val Recall: {val_recall}%')

    training_output.append({
        "epoch": epoch + 1,
        "model_type": model_type,
        "feature_type": feature_type,
        "fusion_type": hyperparameters["fusion_type"],
        "standardize": hyperparameters["standardize"],
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "val_recall": val_recall,
    })

    scheduler.step()

    # =========================
    # 保存 best
    # =========================
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), 'best_model.pth')
        save_feature_config('best_model_config.json', config_to_save)
        print(f"Best model saved at epoch {epoch + 1} with validation accuracy: {val_accuracy}%")




# =========================
# 平均指标
# =========================
avg_val_accuracy = np.mean([x["val_accuracy"] for x in training_output])
avg_val_recall = np.mean([x["val_recall"] for x in training_output])

# =========================
# 保存最终指标
# =========================
save_feature_config(
    f"experiment_metrics_{experiment_name}.json",
    {
        "model_type": model_type,
        "feature_type": feature_type,
        "fusion_type": hyperparameters["fusion_type"],
        "standardize": hyperparameters["standardize"],
        "best_val_accuracy": best_val_accuracy,
        "best_val_recall": max([x["val_recall"] for x in training_output]),
        "avg_val_accuracy": float(avg_val_accuracy),
        "avg_val_recall": float(avg_val_recall),
    },
)
# =========================
# 保存训练过程（给 report.py 用）
# =========================
with open("training_output.json", "w", encoding="utf-8") as f:
    json.dump(training_output, f, indent=4, ensure_ascii=False)

print("training_output.json 已保存")


print("训练完成，指标已保存")