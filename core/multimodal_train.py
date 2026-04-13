import argparse
import copy
import json
import random
import sys
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from transformers import AutoTokenizer

# 兼容两种运行方式：
# 1) 在项目根目录执行：python multimodal_train.py ...
# 2) 在 core 目录执行：python multimodal_train.py ...
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent if CURRENT_DIR.name == "core" else CURRENT_DIR
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.feature_utils import extract_dual_branch_features
from core.lyrics_data_process import compute_repetition_score
from core.model_factory import load_model_and_config
from fusion_models import build_fusion_model
from metrics_utils import DEFAULT_LABELS, accuracy_score, classification_report_dict


class MultimodalSongDataset(Dataset):
    def __init__(self, song_items, label_names):
        self.song_items = song_items
        self.label_to_idx = {name: idx for idx, name in enumerate(label_names)}

    def __len__(self):
        return len(self.song_items)

    def __getitem__(self, idx):
        item = self.song_items[idx]
        return {
            "audio_path": item["processed_audio"],
            "lyrics_path": item.get("processed_lyric"),
            "label": self.label_to_idx[item["genre"]],
            "sample_id": item["sample_id"],
        }


class MultimodalTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_prefix = "multimodal_"
        self.label_names = DEFAULT_LABELS
        self.num_classes = len(self.label_names)

        audio_config_path = self._resolve_input_path(args.audio_config, expected_name="best_model_config.json")
        audio_model_path = self._resolve_input_path(args.audio_model, expected_name="best_model.pth")
        lyrics_config_path = self._resolve_input_path(args.lyrics_config, expected_name="lyrics_best_model_config.json")
        lyrics_model_path = self._resolve_input_path(args.lyrics_model, expected_name="lyrics_best_model.pth")
        self.song_mapping_path = self._resolve_optional_input_path(
            args.song_mapping,
            expected_name="song_mapping.json",
            fallback_dir=PROJECT_ROOT / "dataset_multy2_processed" / "metadata",
        )
        self.audio_root_path = self._resolve_input_path(
            args.audio_root,
            expected_name="audio",
            fallback_dir=PROJECT_ROOT / "dataset_multy2_processed",
        )
        self.lyrics_root_path = self._resolve_input_path(
            args.lyrics_root,
            expected_name="lyrics",
            fallback_dir=PROJECT_ROOT / "dataset_multy2_processed",
        )

        self.output_dir = self._resolve_output_dir(args.output_dir)
        print(f"[Path] audio_config: {audio_config_path}")
        print(f"[Path] audio_model: {audio_model_path}")
        print(f"[Path] lyrics_config: {lyrics_config_path}")
        print(f"[Path] lyrics_model: {lyrics_model_path}")
        print(f"[Path] song_mapping: {self.song_mapping_path}")
        print(f"[Path] audio_root: {self.audio_root_path}")
        print(f"[Path] lyrics_root: {self.lyrics_root_path}")
        print(f"[Path] output_dir: {self.output_dir}")
        print(f"[LR] fusion_lr: {self.args.fusion_lr}")
        print(f"[LR] audio_lr: {self.args.audio_lr}")
        print(f"[LR] lyrics_lr: {self.args.lyrics_lr}")
        print(f"[Train] train_backbones: {self.args.train_backbones}")

        self.audio_model, self.audio_config, _, self.audio_label_mapper = load_model_and_config(
            str(audio_config_path),
            str(audio_model_path),
        )
        self.lyrics_model, self.lyrics_config, _, self.lyrics_label_mapper = load_model_and_config(
            str(lyrics_config_path),
            str(lyrics_model_path),
        )

        if self.audio_model is None or self.lyrics_model is None:
            raise RuntimeError("未能加载音频或歌词模型，请检查权重和配置文件路径。")

        self.base_audio_state = copy.deepcopy(self.audio_model.state_dict())
        self.base_lyrics_state = copy.deepcopy(self.lyrics_model.state_dict())

        self.audio_model.to(self.device).eval()
        self.lyrics_model.to(self.device).eval()
        for param in self.audio_model.parameters():
            param.requires_grad = bool(self.args.train_backbones)
        for param in self.lyrics_model.parameters():
            param.requires_grad = bool(self.args.train_backbones)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.lyrics_config.get("pretrained_model_name", "bert-base-uncased")
        )

        self.song_items = self._load_song_items(self.song_mapping_path) if self.song_mapping_path is not None else []
        if not self.song_items:
            print("[Warn] song_mapping 中样本路径不可用，自动回退到 audio/lyrics 目录扫描配对。")
            self.song_items = self._load_song_items_from_folders(self.audio_root_path, self.lyrics_root_path)
        self.dataset = MultimodalSongDataset(self.song_items, self.label_names)

    def _normalize_label_name(self, label):
        zh_to_en = {
            "布鲁斯": "blues",
            "古典": "classical",
            "乡村": "country",
            "迪斯科": "disco",
            "嘻哈": "hiphop",
            "爵士": "jazz",
            "金属": "metal",
            "流行": "pop",
            "雷鬼": "reggae",
            "摇滚": "rock",
        }
        label = str(label).strip()
        return zh_to_en.get(label, label.lower())

    def _align_probabilities(self, probs, label_mapper=None, config=None):
        if probs.shape[1] == self.num_classes:
            return probs

        labels = []
        if label_mapper is not None and hasattr(label_mapper, "get_labels"):
            labels = label_mapper.get_labels()
        if not labels:
            labels = list((config or {}).get("label_names", []))
        if not labels and (config or {}).get("missing_labels"):
            missing = {self._normalize_label_name(x) for x in (config or {}).get("missing_labels", [])}
            labels = [x for x in self.label_names if x not in missing]
        if not labels:
            labels = self.label_names[: probs.shape[1]]

        target_index = {name: i for i, name in enumerate(self.label_names)}
        aligned = torch.zeros(probs.size(0), self.num_classes, device=probs.device, dtype=probs.dtype)
        mapped = 0
        for source_i in range(min(len(labels), probs.shape[1])):
            key = self._normalize_label_name(labels[source_i])
            if key in target_index:
                aligned[:, target_index[key]] = probs[:, source_i]
                mapped += 1

        if mapped == 0:
            aligned[:, : probs.shape[1]] = probs
        return aligned

    @staticmethod
    def _resolve_output_dir(output_dir):
        raw = Path(output_dir)
        if raw.is_absolute():
            return raw
        if raw == Path("core"):
            return PROJECT_ROOT / "core"
        return (CURRENT_DIR / raw).resolve()

    @staticmethod
    def _resolve_input_path(raw_path, expected_name=None, fallback_dir=None):
        candidate = Path(raw_path)
        candidates = []

        if candidate.is_absolute():
            candidates.append(candidate)
        else:
            candidates.append(candidate)
            candidates.append((CURRENT_DIR / candidate).resolve())
            candidates.append((PROJECT_ROOT / candidate).resolve())
            if candidate.parts and candidate.parts[0] == "core":
                candidates.append((PROJECT_ROOT / Path(*candidate.parts[1:])).resolve())
            if expected_name:
                candidates.append((CURRENT_DIR / expected_name).resolve())
                candidates.append((PROJECT_ROOT / expected_name).resolve())
                candidates.append((PROJECT_ROOT / "core" / expected_name).resolve())
            if fallback_dir is not None and expected_name:
                candidates.append((Path(fallback_dir) / expected_name).resolve())

        for path in candidates:
            if path.exists():
                return path

        checked = "\n".join(f"- {str(p)}" for p in dict.fromkeys(candidates))
        raise FileNotFoundError(
            f"找不到文件: {raw_path}\n已检查路径:\n{checked}"
        )

    @staticmethod
    def _resolve_optional_input_path(raw_path, expected_name=None, fallback_dir=None):
        try:
            return MultimodalTrainer._resolve_input_path(raw_path, expected_name=expected_name, fallback_dir=fallback_dir)
        except FileNotFoundError:
            return None

    @staticmethod
    def _seed_everything(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _load_song_items(self, mapping_path):
        payload = json.loads(Path(mapping_path).read_text(encoding="utf-8"))
        items = []
        for sample_id, meta in payload.items():
            genre = meta["genre"].lower()
            if genre not in self.label_names:
                continue
            audio_ok = Path(meta["processed_audio"]).exists()
            lyric_path = meta.get("processed_lyric")
            lyric_ok = bool(lyric_path) and Path(lyric_path).exists()
            if not audio_ok:
                continue
            items.append(
                {
                    "sample_id": sample_id,
                    "genre": genre,
                    "processed_audio": meta["processed_audio"],
                    "processed_lyric": meta["processed_lyric"] if lyric_ok else None,
                }
            )

        return items

    def _load_song_items_from_folders(self, audio_root, lyrics_root):
        audio_root = Path(audio_root)
        lyrics_root = Path(lyrics_root)
        items = []

        for genre in self.label_names:
            audio_dir = audio_root / genre
            lyrics_dir = lyrics_root / genre
            if not audio_dir.exists() or not lyrics_dir.exists():
                continue

            lyric_files = {p.stem: p for p in lyrics_dir.glob("*.txt")}
            audio_files = {}
            for ext in ("*.wav", "*.mp3", "*.au", "*.flac", "*.ogg"):
                for p in audio_dir.glob(ext):
                    audio_files[p.stem] = p

            for sample_id in sorted(audio_files.keys()):
                items.append(
                    {
                        "sample_id": sample_id,
                        "genre": genre,
                        "processed_audio": str(audio_files[sample_id]),
                        "processed_lyric": str(lyric_files[sample_id]) if sample_id in lyric_files else None,
                    }
                )

        if not items:
            raise RuntimeError(
                "未找到可用多模态样本。请检查:\n"
                f"1) audio目录: {audio_root}\n"
                f"2) lyrics目录: {lyrics_root}\n"
                "3) 是否按相同文件名（不含扩展名）一一对应，例如 blues_00.wav 与 blues_00.txt"
            )
        return items

    def _collate_fn(self, batch):
        mfcc_list, mel_list = [], []
        input_ids_list, attention_mask_list = [], []
        repetition_scores, labels = [], []
        sample_ids = []
        has_lyrics_list = []

        for sample in batch:
            audio, sr = librosa.load(sample["audio_path"], sr=self.audio_config.get("target_sr", 22050))
            mfcc, mel = extract_dual_branch_features(
                audio,
                sr,
                n_mfcc=self.audio_config.get("n_mfcc", 13),
                n_mels=self.audio_config.get("n_mels", 128),
                max_length=self.audio_config.get("max_length", 1000),
                standardize=self.audio_config.get("standardize", True),
            )
            mfcc_list.append(torch.tensor(np.expand_dims(mfcc, -1), dtype=torch.float32))
            mel_list.append(torch.tensor(np.expand_dims(mel, -1), dtype=torch.float32))

            lyrics_text = ""
            has_lyrics = 0.0
            if sample.get("lyrics_path"):
                lyrics_path = Path(sample["lyrics_path"])
                if lyrics_path.exists():
                    lyrics_text = lyrics_path.read_text(encoding="utf-8", errors="ignore")
                    has_lyrics = 1.0
            encoded = self.tokenizer(
                lyrics_text,
                max_length=self.lyrics_config.get("max_length", 128),
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids_list.append(encoded["input_ids"].squeeze(0))
            attention_mask_list.append(encoded["attention_mask"].squeeze(0))
            repetition_scores.append(compute_repetition_score(lyrics_text))
            has_lyrics_list.append(has_lyrics)

            labels.append(sample["label"])
            sample_ids.append(sample["sample_id"])

        return {
            "mfcc": torch.stack(mfcc_list, dim=0),
            "mel": torch.stack(mel_list, dim=0),
            "input_ids": torch.stack(input_ids_list, dim=0),
            "attention_mask": torch.stack(attention_mask_list, dim=0),
            "repetition_score": torch.tensor(repetition_scores, dtype=torch.float32),
            "has_lyrics": torch.tensor(has_lyrics_list, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long),
            "sample_ids": sample_ids,
        }

    def _extract_audio_logits_and_features(self, mfcc, mel):
        model = self.audio_model
        h_mfcc = model.mfcc_branch(mfcc)
        h_mel = model.mel_branch(mel)

        if model.fusion_type == "concat":
            fused = torch.cat([h_mfcc, h_mel], dim=1)
        else:
            gate_input = torch.cat([h_mfcc, h_mel], dim=1)
            gate = model.gate(gate_input)
            fused = gate * h_mfcc + (1.0 - gate) * h_mel

        z_audio = F.mish(model.bn_fc1(model.fusion_fc1(fused)))
        logits = model.classifier(model.dropout1(z_audio))
        return logits, z_audio

    def _extract_lyrics_logits_and_features(self, input_ids, attention_mask, repetition_score):
        outputs = self.lyrics_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            repetition_score=repetition_score,
        )

        cls_emb = outputs["cls_embedding"]
        context_emb = outputs["context_embedding"]
        gate = outputs["gate"]

        gated_cls = cls_emb * gate[:, :1]
        gated_context = context_emb * gate[:, 1:]
        fused_embedding = torch.cat([gated_cls, gated_context], dim=1)
        z_lyrics = self.lyrics_model.fusion(fused_embedding)
        logits = outputs["logits"]
        return logits, z_lyrics

    def _build_fusion_head(self):
        with torch.no_grad():
            sample = self._collate_fn([self.dataset[0]])
            mfcc = sample["mfcc"].to(self.device)
            mel = sample["mel"].to(self.device)
            input_ids = sample["input_ids"].to(self.device)
            attention_mask = sample["attention_mask"].to(self.device)
            repetition_score = sample["repetition_score"].to(self.device)

            _, z_audio = self._extract_audio_logits_and_features(mfcc, mel)
            _, z_lyrics = self._extract_lyrics_logits_and_features(input_ids, attention_mask, repetition_score)

        fusion_model = build_fusion_model(
            fusion=self.args.fusion,
            audio_dim=z_audio.shape[1],
            lyrics_dim=z_lyrics.shape[1],
            num_classes=self.num_classes,
        )
        return fusion_model.to(self.device)

    def _run_epoch(self, loader, fusion_model, criterion, optimizer=None, train=False):
        if train:
            fusion_model.train()
            if self.args.train_backbones:
                self.audio_model.train()
                self.lyrics_model.train()
            else:
                self.audio_model.eval()
                self.lyrics_model.eval()
        else:
            fusion_model.eval()
            self.audio_model.eval()
            self.lyrics_model.eval()

        losses = []
        all_targets, all_preds = [], []
        all_weights = []

        iterator = tqdm(loader, disable=not self.args.verbose)
        for batch in iterator:
            labels = batch["labels"].to(self.device)
            mfcc = batch["mfcc"].to(self.device)
            mel = batch["mel"].to(self.device)
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            repetition_score = batch["repetition_score"].to(self.device)
            has_lyrics = batch["has_lyrics"].to(self.device).unsqueeze(1)

            grad_context = torch.enable_grad() if (train and self.args.train_backbones) else torch.no_grad()
            with grad_context:
                audio_logits, z_audio = self._extract_audio_logits_and_features(mfcc, mel)
                lyrics_logits, z_lyrics = self._extract_lyrics_logits_and_features(input_ids, attention_mask, repetition_score)
                p_audio = torch.softmax(audio_logits, dim=1)
                p_lyrics = torch.softmax(lyrics_logits, dim=1)
                p_audio = self._align_probabilities(p_audio, label_mapper=self.audio_label_mapper, config=self.audio_config)
                p_lyrics = self._align_probabilities(p_lyrics, label_mapper=self.lyrics_label_mapper, config=self.lyrics_config)
                z_lyrics = z_lyrics * has_lyrics

            if self.args.fusion == "concat":
                outputs = fusion_model(z_audio, z_lyrics)
            elif self.args.fusion == "weighting":
                outputs = fusion_model(p_audio, p_lyrics, has_lyrics=has_lyrics)
            elif self.args.fusion == "dynamic":
                outputs = fusion_model(z_audio, z_lyrics, p_audio, p_lyrics, has_lyrics=has_lyrics)
            else:
                raise ValueError(f"Unsupported fusion: {self.args.fusion}")

            logits = outputs["logits"]
            loss = criterion(logits, labels)

            if train and optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            losses.append(loss.item())
            preds = torch.argmax(outputs["probabilities"], dim=1)
            all_targets.extend(labels.detach().cpu().numpy().tolist())
            all_preds.extend(preds.detach().cpu().numpy().tolist())

            if outputs.get("weights") is not None:
                all_weights.extend(outputs["weights"].detach().cpu().numpy().reshape(-1).tolist())

        metrics = classification_report_dict(all_targets, all_preds, self.label_names)
        metrics["accuracy"] = accuracy_score(all_targets, all_preds)
        metrics["loss"] = float(np.mean(losses)) if losses else 0.0
        metrics["targets"] = all_targets
        metrics["predictions"] = all_preds
        metrics["weights"] = all_weights
        return metrics

    def train(self):
        self._seed_everything(self.args.seed)
        out_dir = self.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        kf = KFold(n_splits=3, shuffle=True, random_state=self.args.seed)
        criterion = nn.CrossEntropyLoss()

        training_history = []
        fold_test_metrics = []
        best_val_f1_global = -1.0
        best_state_dict = None

        for fold_idx, (train_val_idx, test_idx) in enumerate(kf.split(range(len(self.dataset))), start=1):
            self.audio_model.load_state_dict(copy.deepcopy(self.base_audio_state))
            self.lyrics_model.load_state_dict(copy.deepcopy(self.base_lyrics_state))
            self.audio_model.to(self.device)
            self.lyrics_model.to(self.device)

            train_val_idx = list(train_val_idx)
            random.Random(self.args.seed + fold_idx).shuffle(train_val_idx)
            split = int(len(train_val_idx) * self.args.train_ratio)
            train_idx = train_val_idx[:split]
            val_idx = train_val_idx[split:]

            train_loader = DataLoader(
                Subset(self.dataset, train_idx),
                batch_size=self.args.batch_size,
                shuffle=True,
                num_workers=self.args.num_workers,
                collate_fn=self._collate_fn,
            )
            val_loader = DataLoader(
                Subset(self.dataset, val_idx),
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=self.args.num_workers,
                collate_fn=self._collate_fn,
            )
            test_loader = DataLoader(
                Subset(self.dataset, list(test_idx)),
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=self.args.num_workers,
                collate_fn=self._collate_fn,
            )

            fusion_model = self._build_fusion_head()
            param_groups = [{"params": fusion_model.parameters(), "lr": self.args.fusion_lr}]
            if self.args.train_backbones:
                param_groups.append({"params": self.audio_model.parameters(), "lr": self.args.audio_lr})
                param_groups.append({"params": self.lyrics_model.parameters(), "lr": self.args.lyrics_lr})
            optimizer = Adam(param_groups)

            best_fold_val_f1 = -1.0
            best_fold_state = None

            for epoch in range(1, self.args.epochs + 1):
                train_metrics = self._run_epoch(train_loader, fusion_model, criterion, optimizer=optimizer, train=True)
                val_metrics = self._run_epoch(val_loader, fusion_model, criterion, optimizer=None, train=False)

                print(
                    f"Fold {fold_idx} | Epoch {epoch:02d} | "
                    f"Train Acc {train_metrics['accuracy'] * 100:.2f}% | "
                    f"Val Acc {val_metrics['accuracy'] * 100:.2f}% | "
                    f"F1 {val_metrics['macro_f1'] * 100:.2f}%"
                )

                training_history.append(
                    {
                        "fold": fold_idx,
                        "epoch": epoch,
                        "train_loss": train_metrics["loss"],
                        "train_accuracy": train_metrics["accuracy"] * 100,
                        "val_loss": val_metrics["loss"],
                        "val_accuracy": val_metrics["accuracy"] * 100,
                        "val_precision": val_metrics["macro_precision"] * 100,
                        "val_recall": val_metrics["macro_recall"] * 100,
                        "val_f1": val_metrics["macro_f1"] * 100,
                    }
                )

                if val_metrics["macro_f1"] > best_fold_val_f1:
                    best_fold_val_f1 = val_metrics["macro_f1"]
                    best_fold_state = {
                        "fusion": {k: v.cpu() for k, v in fusion_model.state_dict().items()},
                        "audio": {k: v.cpu() for k, v in self.audio_model.state_dict().items()},
                        "lyrics": {k: v.cpu() for k, v in self.lyrics_model.state_dict().items()},
                    }

            if best_fold_state is not None:
                fusion_model.load_state_dict(best_fold_state["fusion"])
                if self.args.train_backbones:
                    self.audio_model.load_state_dict(best_fold_state["audio"])
                    self.lyrics_model.load_state_dict(best_fold_state["lyrics"])

            test_metrics = self._run_epoch(test_loader, fusion_model, criterion, optimizer=None, train=False)
            print(
                f"Fold {fold_idx} Test Acc {test_metrics['accuracy'] * 100:.2f}% | "
                f"F1 {test_metrics['macro_f1'] * 100:.2f}%"
            )
            fold_test_metrics.append(
                {
                    "fold": fold_idx,
                    "accuracy": test_metrics["accuracy"],
                    "macro_precision": test_metrics["macro_precision"],
                    "macro_recall": test_metrics["macro_recall"],
                    "macro_f1": test_metrics["macro_f1"],
                    "genre_f1": test_metrics["genre_f1"],
                    "per_class": test_metrics["per_class"],
                    "confusion_matrix": test_metrics["confusion_matrix"],
                    "avg_weight_w": float(np.mean(test_metrics["weights"])) if test_metrics["weights"] else None,
                    "weights": test_metrics["weights"],
                }
            )

            if best_fold_val_f1 > best_val_f1_global and best_fold_state is not None:
                best_val_f1_global = best_fold_val_f1
                best_state_dict = best_fold_state

        history_path = out_dir / f"{self.output_prefix}training_output_{self.args.fusion}.json"
        history_path.write_text(json.dumps(training_history, indent=2, ensure_ascii=False), encoding="utf-8")

        macro_f1s = [x["macro_f1"] for x in fold_test_metrics]
        macro_precs = [x["macro_precision"] for x in fold_test_metrics]
        macro_recs = [x["macro_recall"] for x in fold_test_metrics]

        final_genre_f1 = {name: float(np.mean([f["genre_f1"][name] for f in fold_test_metrics])) for name in self.label_names}
        result_payload = {
            "fusion": self.args.fusion,
            "folds": fold_test_metrics,
            "genre_f1": final_genre_f1,
            "macro_precision": float(np.mean(macro_precs)),
            "macro_recall": float(np.mean(macro_recs)),
            "macro_f1": float(np.mean(macro_f1s)),
            "table_row": {
                "Fusion Method": self.args.fusion.capitalize(),
                **{name.capitalize(): final_genre_f1[name] for name in self.label_names},
                "Macro avg": float(np.mean(macro_f1s)),
            },
        }

        if self.args.fusion == "dynamic":
            all_weights = []
            for fold in fold_test_metrics:
                all_weights.extend(fold.get("weights", []))
            result_payload["avg_weight_w"] = float(np.mean(all_weights)) if all_weights else None

        result_path = out_dir / f"{self.output_prefix}test_results_{self.args.fusion}.json"
        result_path.write_text(json.dumps(result_payload, indent=2, ensure_ascii=False), encoding="utf-8")

        model_path = out_dir / f"{self.output_prefix}best_model_{self.args.fusion}.pth"
        if best_state_dict is not None:
            torch.save(best_state_dict, model_path)

        self._plot_training_curve(training_history, out_dir / f"{self.output_prefix}training_curve_{self.args.fusion}.png")
        cm_total = np.sum(np.array([f["confusion_matrix"] for f in fold_test_metrics]), axis=0)
        self._plot_confusion_matrix(
            cm_total,
            out_dir / f"{self.output_prefix}confusion_matrix_{self.args.fusion}.png",
        )
        self._plot_training_report(
            training_history=training_history,
            confusion_matrix=cm_total,
            avg_test_acc=float(np.mean([f["accuracy"] for f in fold_test_metrics])) * 100.0,
            avg_test_f1=float(np.mean([f["macro_f1"] for f in fold_test_metrics])) * 100.0,
            output_path=out_dir / f"{self.output_prefix}training_report_{self.args.fusion}.png",
        )

        if self.args.fusion == "dynamic":
            all_weights = []
            for fold in fold_test_metrics:
                all_weights.extend(fold.get("weights", []))
            self._plot_weight_hist(all_weights, out_dir / f"{self.output_prefix}weight_distribution_dynamic.png")

        print(f"训练完成，结果已保存到: {out_dir}")

    def _plot_training_curve(self, history, output_path):
        if not history:
            return
        epochs = list(range(1, self.args.epochs + 1))
        folds = sorted({item["fold"] for item in history})

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for fold in folds:
            fold_data = [x for x in history if x["fold"] == fold]
            axes[0].plot(epochs, [x["train_accuracy"] for x in fold_data], label=f"Fold {fold} Train")
            axes[0].plot(epochs, [x["val_accuracy"] for x in fold_data], linestyle="--", label=f"Fold {fold} Val")
            axes[1].plot(epochs, [x["val_f1"] for x in fold_data], label=f"Fold {fold} Val F1")

        axes[0].set_title("Accuracy Curve")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Accuracy (%)")
        axes[0].legend(fontsize=8)

        axes[1].set_title("Validation Macro F1")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("F1 (%)")
        axes[1].legend(fontsize=8)

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close(fig)

    def _plot_confusion_matrix(self, cm, output_path):
        cm = np.asarray(cm)
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, cmap="Blues")
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(self.num_classes))
        ax.set_yticks(range(self.num_classes))
        ax.set_xticklabels(self.label_names, rotation=45, ha="right")
        ax.set_yticklabels(self.label_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix ({self.args.fusion})")

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{cm[i, j]:.1f}", ha="center", va="center", fontsize=8)

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close(fig)

    @staticmethod
    def _plot_weight_hist(weights, output_path):
        if not weights:
            return
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(weights, bins=20, color="#4C72B0", alpha=0.85)
        ax.set_title("Dynamic Fusion Weight Distribution (w)")
        ax.set_xlabel("w")
        ax.set_ylabel("Count")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close(fig)

    def _plot_training_report(self, training_history, confusion_matrix, avg_test_acc, avg_test_f1, output_path):
        if not training_history:
            return

        folds = sorted({item["fold"] for item in training_history})
        fig, axs = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle(f"Training Report (3-Fold Cross Validation) - {self.args.fusion}", fontsize=18)

        def plot_metric(ax, key, title):
            for fold in folds:
                fold_data = [x for x in training_history if x["fold"] == fold]
                epochs = [x["epoch"] for x in fold_data]
                values = [x[key] for x in fold_data]
                ax.plot(epochs, values, label=f"Fold {fold}")
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.legend(fontsize=9)

        plot_metric(axs[0, 0], "train_loss", "Train Loss")
        plot_metric(axs[0, 1], "train_accuracy", "Train Accuracy")
        plot_metric(axs[0, 2], "val_loss", "Validation Loss")
        plot_metric(axs[0, 3], "val_accuracy", "Validation Accuracy")
        plot_metric(axs[1, 0], "val_recall", "Validation Recall")
        plot_metric(axs[1, 1], "val_f1", "Validation F1")

        cm = np.asarray(confusion_matrix)
        im = axs[1, 2].imshow(cm, cmap="viridis")
        axs[1, 2].set_title("Confusion Matrix")
        axs[1, 2].set_xlabel("Predicted")
        axs[1, 2].set_ylabel("True")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                axs[1, 2].text(j, i, f"{int(cm[i, j])}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=axs[1, 2], fraction=0.046, pad=0.04)

        best_val_acc = max(x["val_accuracy"] for x in training_history)
        best_val_f1 = max(x["val_f1"] for x in training_history)

        model_name = self.audio_config.get("model_type", "audio_backbone")
        feature_name = self.audio_config.get("feature_type", "mfcc+mel")
        fusion_name = self.args.fusion
        standardize = self.audio_config.get("standardize", False)
        train_mode = "Fine-tune backbones" if self.args.train_backbones else "Freeze backbones"

        summary_text = (
            f"Model: {model_name}\n"
            f"Feature: {feature_name}\n"
            f"Fusion: {fusion_name}\n"
            f"Standardize: {standardize}\n"
            f"Mode: {train_mode}\n\n"
            f"Fusion LR: {self.args.fusion_lr:.0e}\n"
            f"Audio LR: {self.args.audio_lr:.0e}\n"
            f"Lyrics LR: {self.args.lyrics_lr:.0e}\n\n"
            f"Best Val Acc: {best_val_acc:.2f}%\n"
            f"Best Val F1: {best_val_f1:.2f}%\n\n"
            f"Avg Test Acc: {avg_test_acc:.2f}%\n"
            f"Avg Test F1: {avg_test_f1:.2f}%"
        )
        axs[1, 3].axis("off")
        axs[1, 3].text(
            0.02,
            0.98,
            summary_text,
            va="top",
            fontsize=11,
            bbox=dict(boxstyle="round", facecolor="whitesmoke"),
        )

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(output_path)
        plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Multimodal music genre fusion training")
    parser.add_argument("--fusion", type=str, required=True, choices=["concat", "weighting", "dynamic"])
    parser.add_argument("--song_mapping", type=str, default="dataset_multy2_processed/metadata/song_mapping.json")
    parser.add_argument("--audio_root", type=str, default="dataset_multy2_processed/audio")
    parser.add_argument("--lyrics_root", type=str, default="dataset_multy2_processed/lyrics")
    parser.add_argument("--audio_config", type=str, default="core/best_model_config.json")
    parser.add_argument("--audio_model", type=str, default="core/best_model.pth")
    parser.add_argument("--lyrics_config", type=str, default="core/lyrics_best_model_config.json")
    parser.add_argument("--lyrics_model", type=str, default="core/lyrics_best_model.pth")
    parser.add_argument("--output_dir", type=str, default="core")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--fusion_lr", type=float, default=1e-4)
    parser.add_argument("--audio_lr", type=float, default=5e-5)
    parser.add_argument("--lyrics_lr", type=float, default=5e-4)
    parser.add_argument("--train_backbones", action="store_true", default=True)
    parser.add_argument("--freeze_backbones", action="store_false", dest="train_backbones")
    parser.add_argument("--train_ratio", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    trainer = MultimodalTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()