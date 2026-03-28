import torch
import torch.nn as nn

try:
    from transformers import AutoModel
except ImportError:  # pragma: no cover - runtime dependency
    AutoModel = None


class LyricsGenreBERT(nn.Module):
    """
    创新点：
    1. 双视图池化：融合 [CLS] 与 token 级注意力池化表征，而不是只使用 [CLS]。
    2. 重复感知门控：利用歌词重复度统计动态调节 [CLS] 与上下文语义的融合比例。
    3. 辅助分类头：对纯 [CLS] 分支单独监督，提升主分类头的鲁棒性。
    """

    def __init__(
        self,
        pretrained_model_name="bert-base-uncased",
        num_classes=10,
        dropout=0.3,
        dense_dim=256,
        freeze_encoder=False,
        unfreeze_last_n_layers=0,
        gradient_checkpointing=False,
    ):
        super().__init__()
        if AutoModel is None:
            raise ImportError("transformers is required to use LyricsGenreBERT. Please install transformers.")

        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        if gradient_checkpointing and hasattr(self.bert, "gradient_checkpointing_enable"):
            self.bert.gradient_checkpointing_enable()

        if freeze_encoder:
            for parameter in self.bert.parameters():
                parameter.requires_grad = False
        elif unfreeze_last_n_layers > 0 and hasattr(self.bert, "encoder") and hasattr(self.bert.encoder, "layer"):
            for parameter in self.bert.parameters():
                parameter.requires_grad = False
            for layer in self.bert.encoder.layer[-unfreeze_last_n_layers:]:
                for parameter in layer.parameters():
                    parameter.requires_grad = True
            if hasattr(self.bert, "pooler") and self.bert.pooler is not None:
                for parameter in self.bert.pooler.parameters():
                    parameter.requires_grad = True

        hidden_size = self.bert.config.hidden_size
        self.attention_vector = nn.Linear(hidden_size, 1)
        self.repetition_gate = nn.Sequential(
            nn.Linear(hidden_size * 2 + 1, dense_dim),
            nn.GELU(),
            nn.Linear(dense_dim, 2),
            nn.Sigmoid(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, dense_dim),
            nn.LayerNorm(dense_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(dense_dim, num_classes)
        self.aux_classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def _attention_pool(self, sequence_output, attention_mask):
        scores = self.attention_vector(sequence_output).squeeze(-1)
        scores = scores.masked_fill(attention_mask == 0, -1e4)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(sequence_output * weights.unsqueeze(-1), dim=1)
        return pooled, weights

    def forward(self, input_ids, attention_mask, repetition_score=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        cls_embedding = sequence_output[:, 0, :]
        pooled_context, attention_weights = self._attention_pool(sequence_output, attention_mask)

        if repetition_score is None:
            repetition_score = torch.zeros(cls_embedding.size(0), 1, device=cls_embedding.device)
        elif repetition_score.dim() == 1:
            repetition_score = repetition_score.unsqueeze(-1)

        gate_input = torch.cat([cls_embedding, pooled_context, repetition_score], dim=1)
        gate = self.repetition_gate(gate_input)
        gated_cls = cls_embedding * gate[:, :1]
        gated_context = pooled_context * gate[:, 1:]
        fused_embedding = torch.cat([gated_cls, gated_context], dim=1)

        logits = self.classifier(self.fusion(fused_embedding))
        aux_logits = self.aux_classifier(cls_embedding)
        return {
            "logits": logits,
            "aux_logits": aux_logits,
            "attention_weights": attention_weights,
            "gate": gate,
            "cls_embedding": cls_embedding,
            "context_embedding": pooled_context,
        }