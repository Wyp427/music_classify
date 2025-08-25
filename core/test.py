import torch
from cnn import AudioCNN  # 假设你的模型是定义在 cnn.py 中
from pre_process import preprocess_and_predict  # 假设你的函数在 pre_process.py 中
from label_mapper import GTZANLabelMapper  # 导入你的 GTZANLabelMapper 类

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
model = AudioCNN()  # 确保使用与你训练时相同的模型结构
# 根据设备选择加载模型的位置
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.to(device)  # 将模型移动到相应的设备上
model.eval()  # 切换到评估模式

# 输入音频文件路径
file_path = './datasets/music/blues/blues.00000.au'

# 初始化 GTZANLabelMapper 类
label_mapper = GTZANLabelMapper()

# 调用函数进行预测
predicted_class, probabilities = preprocess_and_predict(model, file_path)

# 输出预测结果
if predicted_class is not None:
    print(f"Predicted class index: {predicted_class}")

    # 获取预测的中文标签
    predicted_label = label_mapper.get_label(predicted_class)
    print(f"Predicted label: {predicted_label}")

    # 输出每个类别的概率，格式：0-概率, 1-概率, ...
    for i, prob in enumerate(probabilities):
        label = label_mapper.get_label(i)  # 获取中文标签
        print(f"{label}-{prob:.4f}")
else:
    print("Error in prediction.")