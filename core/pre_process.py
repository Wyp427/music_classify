import librosa
import numpy as np
import torch

from feature_utils import extract_audio_features, extract_dual_branch_features

#预处理特征

def _to_tensor(features, device):
    features = np.expand_dims(features, axis=-1)
    features = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    return features.to(device)


def _predict_from_audio(
    model,
    audio,
    sr,
    feature_type="mfcc",
    n_mfcc=13,
    n_mels=128,
    max_length=1000,
    standardize=False,
):
    features = extract_audio_features(
        audio,
        sr,
        feature_type=feature_type,
        n_mfcc=n_mfcc,
        n_mels=n_mels,
        max_length=max_length,
        standardize=standardize,
    )

    device = next(model.parameters()).device
    features = _to_tensor(features, device)

    with torch.no_grad():
        prediction = model(features)

    predicted_class = torch.argmax(prediction, dim=1).item()
    probabilities = torch.nn.functional.softmax(prediction, dim=1).squeeze(0).cpu().numpy()
    return predicted_class, probabilities


def _predict_from_audio_dual_branch(
    model,
    audio,
    sr,
    n_mfcc=13,
    n_mels=128,
    max_length=1000,
    standardize=True,
):
    mfcc_features, mel_features = extract_dual_branch_features(
        audio,
        sr,
        n_mfcc=n_mfcc,
        n_mels=n_mels,
        max_length=max_length,
        standardize=standardize,
    )

    device = next(model.parameters()).device
    mfcc_tensor = _to_tensor(mfcc_features, device)
    mel_tensor = _to_tensor(mel_features, device)

    with torch.no_grad():
        prediction = model(mfcc_tensor, mel_tensor)

    predicted_class = torch.argmax(prediction, dim=1).item()
    probabilities = torch.nn.functional.softmax(prediction, dim=1).squeeze(0).cpu().numpy()
    return predicted_class, probabilities


def preprocess_and_predict_file(
    model,
    music_file,
    target_sr=22050,
    n_mfcc=13,
    n_mels=128,
    max_length=1000,
    feature_type="mfcc",
    model_type="single",
    standardize=False,
):
    try:
        music_file.seek(0)
        audio, sr = librosa.load(music_file, sr=target_sr)
        if model_type == "dual_branch":
            return _predict_from_audio_dual_branch(
                model,
                audio,
                sr,
                n_mfcc=n_mfcc,
                n_mels=n_mels,
                max_length=max_length,
                standardize=standardize,
            )
        return _predict_from_audio(
            model,
            audio,
            sr,
            feature_type=feature_type,
            n_mfcc=n_mfcc,
            n_mels=n_mels,
            max_length=max_length,
            standardize=standardize,
        )
    except Exception as e:
        print(f"Error processing the audio file: {e}")
        return None, None


def preprocess_and_predict(
    model,
    file_path,
    target_sr=22050,
    n_mfcc=13,
    n_mels=128,
    max_length=1000,
    feature_type="mfcc",
    model_type="single",
    standardize=False,
):
    try:
        audio, sr = librosa.load(file_path, sr=target_sr)
        if model_type == "dual_branch":
            return _predict_from_audio_dual_branch(
                model,
                audio,
                sr,
                n_mfcc=n_mfcc,
                n_mels=n_mels,
                max_length=max_length,
                standardize=standardize,
            )
        return _predict_from_audio(
            model,
            audio,
            sr,
            feature_type=feature_type,
            n_mfcc=n_mfcc,
            n_mels=n_mels,
            max_length=max_length,
            standardize=standardize,
        )
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None