

"""
model_factory.py  文件说明

该文件用于统一管理模型的构建与加载流程，避免在训练、测试、推理和部署阶段
重复编写模型初始化代码。文件根据配置参数自动判断当前使用的是单分支模型
还是双分支融合模型，并完成对应模型结构的创建、权重加载和运行设备设置。

其中：
1. build_model_from_config 用于根据配置字典动态构建模型；
2. load_model_and_config 用于同时读取模型配置文件和训练好的权重文件，
   并返回可直接用于推理或测试的模型对象。

该文件的作用是保证训练阶段与推理阶段使用一致的模型结构和参数配置，
提高系统的可维护性与实验复现性。
"""

import json
from pathlib import Path

try:
    import torch
except ImportError:  # pragma: no cover - runtime dependency
    torch = None

from cnn import AudioCNN
from dual_branch_cnn import DualBranchFusionCNN
from feature_utils import DEFAULT_INFERENCE_CONFIG, get_input_channels, load_feature_config
from label_mapper import LabelMapper
from lyrics_model import LyricsGenreBERT


DEFAULT_TEXT_CONFIG = {
    "task_type": "lyrics",
    "pretrained_model_name": "bert-base-uncased",
    "max_length": 256,
    "num_classes": 10,
    "dropout": 0.3,
    "dense_dim": 256,
}


def build_model_from_config(config):
    task_type = config.get("task_type", "audio")
    model_type = config.get("model_type", "single")
    num_classes = config.get("num_classes", 10)

    if task_type == "lyrics":
        return LyricsGenreBERT(
            pretrained_model_name=config.get("pretrained_model_name", "bert-base-uncased"),
            num_classes=num_classes,
            dropout=config.get("dropout", 0.3),
            dense_dim=config.get("dense_dim", 256),
        )

    if model_type == "dual_branch":
        return DualBranchFusionCNN(
            num_classes=num_classes,
            mfcc_channels=config.get("n_mfcc", 13),
            mel_channels=config.get("n_mels", 128),
            fusion_type=config.get("fusion_type", "concat"),
        )

    input_channels = config.get(
        "input_channels",
        get_input_channels(
            config.get("feature_type", "mfcc"),
            n_mfcc=config.get("n_mfcc", 13),
            n_mels=config.get("n_mels", 128),
        ),
    )
    return AudioCNN(num_classes=num_classes, input_channels=input_channels)


def _load_label_mapper(config):
    label_names = config.get("label_names")
    mapping_path = config.get("label_mapping_path")
    if not label_names and mapping_path and Path(mapping_path).exists():
        payload = json.loads(Path(mapping_path).read_text(encoding="utf-8"))
        label_names = payload.get("labels")
    return LabelMapper(label_names)


def load_model_and_config(config_path="best_model_config.json", model_path="best_model.pth"):
    config_path = Path(config_path)
    default_config = DEFAULT_TEXT_CONFIG if "lyrics" in config_path.stem else DEFAULT_INFERENCE_CONFIG
    config = load_feature_config(config_path, default=default_config)
    label_mapper = _load_label_mapper(config)

    if torch is None:
        return None, config, None, label_mapper

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_config(config)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, config, device, label_mapper