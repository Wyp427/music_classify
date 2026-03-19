import torch

from cnn import AudioCNN
from feature_utils import load_feature_config
from label_mapper import GTZANLabelMapper
from pre_process import preprocess_and_predict

"""
本文件用于单音频测试与模型推理验证。

核心改动是引入特征配置文件 best_model_config.json，
在测试阶段自动加载训练时使用的特征参数，
避免因特征不一致导致的输入维度错误。

模型构建时根据配置动态设置 input_channels，
保证网络结构与训练阶段一致。

在预测过程中显式传入 target_sr、n_mfcc、n_mels、max_length、
feature_type 等参数，确保特征提取方式完全一致。

该模块实现了测试流程的自动化配置加载，使模型具备良好的可复现性与可迁移性。
"""



# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = load_feature_config("best_model_config.json")

# 加载模型
model = AudioCNN(
    num_classes=config["num_classes"],
    input_channels=config["input_channels"],
)
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.to(device)
model.eval()

# 输入音频文件路径
file_path = './datasets/music/blues/blues.00000.au'

label_mapper = GTZANLabelMapper()

predicted_class, probabilities = preprocess_and_predict(
    model,
    file_path,
    target_sr=config["target_sr"],
    n_mfcc=config["n_mfcc"],
    n_mels=config["n_mels"],
    max_length=config["max_length"],
    feature_type=config["feature_type"],
)

if predicted_class is not None:
    print(f"Feature type: {config['feature_type']}")
    print(f"Predicted class index: {predicted_class}")
    predicted_label = label_mapper.get_label(predicted_class)
    print(f"Predicted label: {predicted_label}")

    for i, prob in enumerate(probabilities):
        label = label_mapper.get_label(i)
        print(f"{label}-{prob:.4f}")
else:
    print("Error in prediction.")