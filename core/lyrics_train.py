import json
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.model_selection import KFold

from lyrics_data_process import load_lyrics_dataset, save_label_mapping, DEFAULT_EXPECTED_LABELS
from lyrics_model import LyricsGenreBERT


# =========================
# 配置（保留你的完整版本）
# =========================
config = {
    "task_type": "lyrics",
    "dataset_path": r"D:\music_classify_project\dataset_multy2_processed\lyrics",

    "pretrained_model_name": "bert-base-uncased",
    "max_length": 128,
    "batch_size": 4,

    "learning_rate": 5e-7,   #
    "num_epochs": 20,

    "dropout": 0.3,
    "dense_dim": 128,

    "aux_loss_weight": 0.2,

    "freeze_encoder": True,
    "unfreeze_last_n_layers": 0,

    "gradient_checkpointing": False,
    "num_workers": 0,

    "expected_labels": DEFAULT_EXPECTED_LABELS,

    "random_seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


torch.manual_seed(config["random_seed"])
np.random.seed(config["random_seed"])

device = torch.device(config["device"])
print(f"使用设备: {device}")


# =========================
# Dataset
# =========================
class LyricsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# =========================
# 指标
# =========================
def compute_macro_recall(targets, preds, num_classes):
    recalls = []
    targets = np.array(targets)
    preds = np.array(preds)

    for c in range(num_classes):
        mask = targets == c
        if mask.sum() == 0:
            continue
        tp = ((preds == c) & mask).sum()
        recalls.append(tp / mask.sum())

    return float(np.mean(recalls))


def compute_macro_f1(targets, preds, num_classes):
    f1s = []
    targets = np.array(targets)
    preds = np.array(preds)

    for c in range(num_classes):
        tp = ((preds == c) & (targets == c)).sum()
        fp = ((preds == c) & (targets != c)).sum()
        fn = ((preds != c) & (targets == c)).sum()

        if tp == 0:
            f1s.append(0)
            continue

        p = tp / (tp + fp + 1e-8)
        r = tp / (tp + fn + 1e-8)
        f1 = 2 * p * r / (p + r + 1e-8)
        f1s.append(f1)

    return float(np.mean(f1s))


# =========================
# 加载数据
# =========================
dataset = load_lyrics_dataset(config["dataset_path"])
tokenizer = AutoTokenizer.from_pretrained(config["pretrained_model_name"])

# ⭐修复调用顺序
save_label_mapping(dataset["label_names"], "lyrics_label_mapping.json")

kf = KFold(n_splits=3, shuffle=True, random_state=config["random_seed"])

training_output = []
fold_results = []


# =========================
# 3折训练
# =========================
for fold, (train_idx, test_idx) in enumerate(kf.split(dataset["texts"])):

    print(f"\n===== Fold {fold+1} =====")

    train_idx = list(train_idx)
    np.random.shuffle(train_idx)

    split = int(0.85 * len(train_idx))
    train_sub_idx = train_idx[:split]
    val_sub_idx = train_idx[split:]

    train_dataset = LyricsDataset(
        [dataset["texts"][i] for i in train_sub_idx],
        [dataset["labels"][i] for i in train_sub_idx],
        tokenizer,
        config["max_length"],
    )

    val_dataset = LyricsDataset(
        [dataset["texts"][i] for i in val_sub_idx],
        [dataset["labels"][i] for i in val_sub_idx],
        tokenizer,
        config["max_length"],
    )

    test_dataset = LyricsDataset(
        [dataset["texts"][i] for i in test_idx],
        [dataset["labels"][i] for i in test_idx],
        tokenizer,
        config["max_length"],
    )

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    model = LyricsGenreBERT(
        pretrained_model_name=config["pretrained_model_name"],
        num_classes=len(dataset["label_names"]),
        dropout=config["dropout"],
        dense_dim=config["dense_dim"],
        freeze_encoder=config["freeze_encoder"],
        unfreeze_last_n_layers=config["unfreeze_last_n_layers"],
        gradient_checkpointing=config["gradient_checkpointing"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_loader) * config["num_epochs"],
    )

    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0

    # =========================
    # epoch
    # =========================
    for epoch in range(config["num_epochs"]):

        model.train()
        train_loss = 0
        train_correct = 0
        total = 0

        for batch in tqdm(train_loader,disable=True):

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            logits = outputs["logits"]
            aux_logits = outputs["aux_logits"]

            loss_main = criterion(logits, labels)
            loss_aux = criterion(aux_logits, labels)

            loss = loss_main + config["aux_loss_weight"] * loss_aux

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = 100 * train_correct / total
        train_loss /= len(train_loader)

        # ===== 验证 =====
        model.eval()
        val_targets, val_preds = [], []
        val_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

                logits = outputs["logits"]
                aux_logits = outputs["aux_logits"]

                loss_main = criterion(logits, labels)
                loss_aux = criterion(aux_logits, labels)

                loss = loss_main + config["aux_loss_weight"] * loss_aux

                val_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                val_targets.extend(labels.cpu().numpy())
                val_preds.extend(preds.cpu().numpy())

        val_loss /= len(val_loader)
        val_acc = 100 * np.mean(np.array(val_preds) == np.array(val_targets))
        val_recall = 100 * compute_macro_recall(val_targets, val_preds, len(dataset["label_names"]))
        val_f1 = 100 * compute_macro_f1(val_targets, val_preds, len(dataset["label_names"]))

        print(f"Epoch {epoch+1} | Train Acc {train_acc:.2f}% | Val Acc {val_acc:.2f}% | F1 {val_f1:.2f}%")

        training_output.append({
            "epoch": epoch+1,
            "learning_rate": config["learning_rate"],
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_recall": val_recall,
            "val_f1": val_f1
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "lyrics_best_model.pth")

    # =========================
    # Test
    # =========================
    model.eval()
    test_targets, test_preds = [], []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]

            preds = torch.argmax(logits, dim=1)

            test_targets.extend(labels.cpu().numpy().tolist())
            test_preds.extend(preds.cpu().numpy().tolist())

    test_acc = 100 * np.mean(np.array(test_preds) == np.array(test_targets))
    test_f1 = 100 * compute_macro_f1(test_targets, test_preds, len(dataset["label_names"]))

    print(f"Fold {fold+1} Test Acc {test_acc:.2f}% | F1 {test_f1:.2f}%")

    fold_results.append((test_acc, test_f1))


# =========================
# 保存
# =========================
with open("lyrics_training_output.json", "w", encoding="utf-8") as f:
    json.dump(training_output, f, indent=4)

with open("lyrics_test_results.json", "w") as f:
    json.dump({
        "test_accs": [float(x[0]) for x in fold_results],
        "test_f1s": [float(x[1]) for x in fold_results]
    }, f)

with open("lyrics_test_predictions.json", "w") as f:
    json.dump({
        "targets": test_targets,
        "predictions": test_preds
    }, f)

print("歌词模型训练完成")