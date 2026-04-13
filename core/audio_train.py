import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset, Subset
from tqdm import tqdm
import json
from sklearn.model_selection import KFold

from data_process import audio_to_cnn_data
from feature_utils import get_input_channels, save_feature_config
from model_factory import build_model_from_config


# =========================
# 超参数配置
# =========================
hyperparameters = {
    # ===== 数据 =====
    "folder_path": "D:/music_classify_project/dataset_multy2_processed/audio",

    # ===== 模型 =====
    "model_type": "dual_branch",
    "feature_type": "mfcc_mel",
    "fusion_type": "gated",
    "standardize": True,

    # ===== 特征 =====
    "n_mfcc": 13,
    "n_mels": 128,
    "max_length": 1000,

    # ===== 训练 =====
    "batch_size": 16,
    "learning_rate":5e-5,   #选择不同学习率进行消融实验
    "num_epochs": 20,

    # ===== 优化 =====
    "weight_decay": 5e-4,

    # ===== 学习率调度 =====
    "step_size": 20,
    "gamma": 0.678,

    # ===== 其他 =====
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "random_seed": 42,
}


# =========================
# 指标
# =========================
def compute_macro_recall(targets, predictions, num_classes):
    recalls = []
    targets = np.array(targets)
    predictions = np.array(predictions)
    for c in range(num_classes):
        mask = targets == c
        if mask.sum() == 0:
            continue
        tp = ((predictions == c) & mask).sum()
        recalls.append(tp / mask.sum())
    return np.mean(recalls)


def compute_macro_f1(targets, predictions, num_classes):
    f1s = []
    targets = np.array(targets)
    predictions = np.array(predictions)
    for c in range(num_classes):
        tp = ((predictions == c) & (targets == c)).sum()
        fp = ((predictions == c) & (targets != c)).sum()
        fn = ((predictions != c) & (targets == c)).sum()

        if tp == 0:
            f1s.append(0)
            continue

        p = tp / (tp + fp + 1e-8)
        r = tp / (tp + fn + 1e-8)
        f1 = 2 * p * r / (p + r + 1e-8)
        f1s.append(f1)
    return np.mean(f1s)


def compute_classification_metrics(targets, predictions, label_names):
    targets = np.array(targets)
    predictions = np.array(predictions)
    num_classes = len(label_names)

    genre_f1 = {}
    precisions = []
    recalls = []
    f1s = []

    for c, label in enumerate(label_names):
        tp = ((predictions == c) & (targets == c)).sum()
        fp = ((predictions == c) & (targets != c)).sum()
        fn = ((predictions != c) & (targets == c)).sum()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        precisions.append(float(precision))
        recalls.append(float(recall))
        f1s.append(float(f1))
        genre_f1[label] = float(f1)

    return {
        "genre_f1": genre_f1,
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
    }


# =========================
# 选择Adam优化器
# =========================
def create_optimizer(model):
    return optim.Adam(
        model.parameters(),
        lr=hyperparameters["learning_rate"],
        weight_decay=hyperparameters["weight_decay"],
        betas=(0.9, 0.999)
    )


# =========================
# 随机种子
# =========================
torch.manual_seed(hyperparameters["random_seed"])
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

    dataset = TensorDataset(
        torch.from_numpy(mfcc_data).float(),
        torch.from_numpy(mel_data).float(),
        torch.from_numpy(encoded_labels).long()
    )
else:
    raise NotImplementedError("当前只保留 dual_branch 版本")

num_classes = len(np.unique(encoded_labels))
label_names = [
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


# =========================
# 三折交叉验证
# =========================
kf = KFold(n_splits=3, shuffle=True, random_state=42)

training_output = []
fold_results = []

for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):

    print(f"\n===== Fold {fold+1} =====")

    # 关键：打乱
    train_idx = list(train_idx)
    np.random.shuffle(train_idx)

    split = int(0.85 * len(train_idx))
    train_sub_idx = train_idx[:split]
    val_sub_idx = train_idx[split:]

    train_dataset = Subset(dataset, train_sub_idx)
    val_dataset = Subset(dataset, val_sub_idx)
    test_dataset = Subset(dataset, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # =========================
    # 模型
    # =========================
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

    model = build_model_from_config(config_to_save).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model)

    scheduler = StepLR(optimizer,
                       step_size=hyperparameters["step_size"],
                       gamma=hyperparameters["gamma"])

    best_val_accuracy = 0

    # =========================
    # 训练
    # =========================
    for epoch in range(hyperparameters["num_epochs"]):

        model.train()
        train_loss_total = 0
        train_correct = 0
        train_total = 0

        for mfcc, mel, target in train_loader:
            mfcc, mel, target = mfcc.to(device), mel.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(mfcc, mel)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item()

            _, pred = output.max(1)
            train_total += target.size(0)
            train_correct += pred.eq(target).sum().item()

        train_loss = train_loss_total / len(train_loader)
        train_acc = 100. * train_correct / train_total

        # ===== 验证 =====
        model.eval()
        val_targets, val_preds = [], []
        val_loss_total = 0

        with torch.no_grad():
            for mfcc, mel, target in val_loader:
                mfcc, mel = mfcc.to(device), mel.to(device)
                output = model(mfcc, mel)

                loss = criterion(output, target.to(device))
                val_loss_total += loss.item()

                _, pred = output.max(1)
                val_targets.extend(target.numpy())
                val_preds.extend(pred.cpu().numpy())

        val_loss = val_loss_total / len(val_loader)
        val_acc = 100. * np.mean(np.array(val_preds) == np.array(val_targets))
        val_recall = 100. * compute_macro_recall(val_targets, val_preds, num_classes)
        val_f1 = 100. * compute_macro_f1(val_targets, val_preds, num_classes)

        print(f"Epoch {epoch+1} | Train Acc {train_acc:.2f}% | Val Acc {val_acc:.2f}% | F1 {val_f1:.2f}%")

        training_output.append({
            "epoch": epoch+1,
            "learning_rate": hyperparameters["learning_rate"],
            "model_type": model_type,
            "feature_type": feature_type,
            "fusion_type": hyperparameters["fusion_type"],
            "standardize": hyperparameters["standardize"],
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_recall": val_recall,
            "val_f1": val_f1
        })

        scheduler.step()

        # 保存best
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            save_feature_config('best_model_config.json', config_to_save)

    # =========================
    # Test
    # =========================
    model.eval()
    test_targets, test_preds = [], []

    with torch.no_grad():
        for mfcc, mel, target in test_loader:
            mfcc, mel = mfcc.to(device), mel.to(device)
            output = model(mfcc, mel)
            _, pred = output.max(1)

            test_targets.extend(target.numpy())
            test_preds.extend(pred.cpu().numpy())

    test_acc = 100. * np.mean(np.array(test_preds) == np.array(test_targets))
    test_f1 = 100. * compute_macro_f1(test_targets, test_preds, num_classes)

    print(f"Fold {fold+1} Test Acc {test_acc:.2f}% | F1 {test_f1:.2f}%")

    fold_metrics = compute_classification_metrics(test_targets, test_preds, label_names)
    fold_results.append({
        "accuracy": float(test_acc),
        "f1": float(test_f1),
        "genre_f1": fold_metrics["genre_f1"],
        "macro_precision": fold_metrics["macro_precision"],
        "macro_recall": fold_metrics["macro_recall"],
        "macro_f1": fold_metrics["macro_f1"],
    })

    # 修复JSON
    with open("test_predictions.json", "w") as f:
        json.dump({
            "targets": [int(x) for x in test_targets],
            "predictions": [int(x) for x in test_preds]
        }, f)


# =========================
# 保存训练过程
# =========================
with open("training_output.json", "w", encoding="utf-8") as f:
    json.dump(training_output, f, indent=4, ensure_ascii=False)

print("training_output.json 已保存")

# =========================
# 保存 test 结果
# =========================
with open("test_results.json", "w") as f:
    final_genre_f1 = {
        name: float(np.mean([fold["genre_f1"][name] for fold in fold_results]))
        for name in label_names
    }
    macro_precision = float(np.mean([fold["macro_precision"] for fold in fold_results]))
    macro_recall = float(np.mean([fold["macro_recall"] for fold in fold_results]))
    macro_f1 = float(np.mean([fold["macro_f1"] for fold in fold_results]))
    payload = {
        "test_accs": [fold["accuracy"] for fold in fold_results],
        "test_f1s": [fold["f1"] for fold in fold_results],
        "genre_f1": final_genre_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "table_row": {
            "Fusion Method": hyperparameters["fusion_type"].capitalize(),
            **{name.capitalize(): final_genre_f1[name] for name in label_names},
            "Macro avg": macro_f1,
        },
    }
    json.dump(payload, f, indent=4, ensure_ascii=False)

print("test_results.json 已保存")

# =========================
# 保存最终指标
# =========================
accs = [x["accuracy"] for x in fold_results]
f1s = [x["f1"] for x in fold_results]

save_feature_config(
    f"experiment_metrics_{experiment_name}.json",
    {
        "learning_rate": hyperparameters["learning_rate"],
        "avg_accuracy": float(np.mean(accs)),
        "avg_f1": float(np.mean(f1s)),
        "std_accuracy": float(np.std(accs)),
        "std_f1": float(np.std(f1s)),
        "fold_accuracies": accs,
        "fold_f1_scores": f1s
    },
)

print("训练完成，指标已保存")