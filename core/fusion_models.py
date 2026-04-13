import torch
import torch.nn as nn
import torch.nn.functional as F


class ConcatFusionHead(nn.Module):
    """特征级融合：concat(z_audio, z_lyrics) -> MLP -> logits"""

    def __init__(self, audio_dim: int, lyrics_dim: int, num_classes: int = 10, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(audio_dim + lyrics_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, z_audio: torch.Tensor, z_lyrics: torch.Tensor):
        z = torch.cat([z_audio, z_lyrics], dim=1)
        logits = self.classifier(z)
        return {
            "logits": logits,
            "probabilities": torch.softmax(logits, dim=1),
            "weights": None,
        }


class WeightingFusionHead(nn.Module):
    """决策级融合：p = 0.5 * p_audio + 0.5 * p_lyrics"""

    def forward(self, p_audio: torch.Tensor, p_lyrics: torch.Tensor, has_lyrics: torch.Tensor = None):
        p_final = 0.5 * p_audio + 0.5 * p_lyrics
        if has_lyrics is not None:
            # has_lyrics: [B, 1], 1表示有歌词，0表示无歌词
            p_final = has_lyrics * p_final + (1.0 - has_lyrics) * p_audio
        p_final = torch.clamp(p_final, min=1e-8, max=1.0)
        logits = torch.log(p_final)
        return {
            "logits": logits,
            "probabilities": p_final,
            "weights": None,
        }


class DynamicFusionHead(nn.Module):
    """动态融合：w = sigmoid(MLP([z_audio; z_lyrics]))，p = w*p_audio + (1-w)*p_lyrics"""

    def __init__(self, audio_dim: int, lyrics_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.weight_net = nn.Sequential(
            nn.Linear(audio_dim + lyrics_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        z_audio: torch.Tensor,
        z_lyrics: torch.Tensor,
        p_audio: torch.Tensor,
        p_lyrics: torch.Tensor,
        has_lyrics: torch.Tensor = None,
    ):
        z = torch.cat([z_audio, z_lyrics], dim=1)
        w = torch.sigmoid(self.weight_net(z))
        p_final = w * p_audio + (1.0 - w) * p_lyrics
        if has_lyrics is not None:
            p_final = has_lyrics * p_final + (1.0 - has_lyrics) * p_audio
            w = has_lyrics * w + (1.0 - has_lyrics) * torch.ones_like(w)
        p_final = torch.clamp(p_final, min=1e-8, max=1.0)
        logits = torch.log(p_final)
        return {
            "logits": logits,
            "probabilities": p_final,
            "weights": w,
        }


def build_fusion_model(fusion: str, audio_dim: int, lyrics_dim: int, num_classes: int = 10):
    fusion = fusion.lower()
    if fusion == "concat":
        return ConcatFusionHead(audio_dim=audio_dim, lyrics_dim=lyrics_dim, num_classes=num_classes)
    if fusion == "weighting":
        return WeightingFusionHead()
    if fusion == "dynamic":
        return DynamicFusionHead(audio_dim=audio_dim, lyrics_dim=lyrics_dim)
    raise ValueError(f"Unsupported fusion: {fusion}")