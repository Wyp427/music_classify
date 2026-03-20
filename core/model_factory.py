import torch

from cnn import AudioCNN
from dual_branch_cnn import DualBranchFusionCNN
from feature_utils import DEFAULT_INFERENCE_CONFIG, get_input_channels, load_feature_config

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



def build_model_from_config(config):
    model_type = config.get("model_type", "single")
    num_classes = config.get("num_classes", 10)

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


def load_model_and_config(config_path="best_model_config.json", model_path="best_model.pth"):
    config = load_feature_config(config_path, default=DEFAULT_INFERENCE_CONFIG)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_config(config)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, config, device