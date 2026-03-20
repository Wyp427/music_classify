from model_factory import load_model_and_config
from label_mapper import GTZANLabelMapper
from pre_process import preprocess_and_predict

#测试代码

model, config, _ = load_model_and_config("best_model_config.json", "best_model.pth")
file_path = './datasets/music/blues/blues.00000.au'
label_mapper = GTZANLabelMapper()

predicted_class, probabilities = preprocess_and_predict(
    model,
    file_path,
    target_sr=config["target_sr"],
    n_mfcc=config["n_mfcc"],
    n_mels=config["n_mels"],
    max_length=config["max_length"],
    feature_type=config.get("feature_type", "mfcc"),
    model_type=config.get("model_type", "single"),
    standardize=config.get("standardize", False),
)

if predicted_class is not None:
    print(f"Model type: {config.get('model_type', 'single')}")
    print(f"Feature type: {config.get('feature_type', 'mfcc')}")
    print(f"Predicted class index: {predicted_class}")
    predicted_label = label_mapper.get_label(predicted_class)
    print(f"Predicted label: {predicted_label}")

    for i, prob in enumerate(probabilities):
        label = label_mapper.get_label(i)
        print(f"{label}-{prob:.4f}")
else:
    print("Error in prediction.")