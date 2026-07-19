"""Hybrid global-vector and token-sequence attack model."""

from __future__ import annotations

from .base import AttackBatch, AttackOutput
from .torch_base import TorchAttackModel, nn, torch


if nn is not None:
    class HybridAttack(TorchAttackModel):
        def __init__(self, config) -> None:
            super().__init__(config)
            global_hidden = config.model_kwargs.get("global_hidden_dim", 128)
            token_hidden = config.model_kwargs.get("token_hidden_dim", 96)
            fusion_hidden = config.model_kwargs.get("fusion_hidden_dim", 128)

            self.global_encoder = nn.Sequential(
                nn.Linear(config.global_feature_dim, global_hidden),
                nn.LayerNorm(global_hidden),
                nn.GELU(),
                nn.Dropout(config.dropout),
            )
            self.token_encoder = nn.LSTM(
                input_size=config.token_feature_dim,
                hidden_size=token_hidden,
                batch_first=True,
                bidirectional=True,
            )
            self.token_attention = nn.Linear(token_hidden * 2, 1)
            self.classifier = nn.Sequential(
                nn.Linear(global_hidden + token_hidden * 2, fusion_hidden),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(fusion_hidden, 1),
            )

        def forward(self, batch: AttackBatch) -> AttackOutput:
            if batch.global_features is None or batch.token_features is None:
                raise ValueError(
                    "HybridAttack requires global and token features."
                )
            global_embedding = self.global_encoder(batch.global_features)
            token_hidden, _ = self.token_encoder(batch.token_features)
            attn_logits = self.token_attention(token_hidden).squeeze(-1)
            if batch.attention_mask is not None:
                attn_logits = attn_logits.masked_fill(
                    ~batch.attention_mask.bool(), float("-inf")
                )
            weights = torch.softmax(attn_logits, dim=-1)
            token_embedding = (
                token_hidden * weights.unsqueeze(-1)
            ).sum(dim=1)
            fused = torch.cat([global_embedding, token_embedding], dim=-1)
            logits = self.classifier(fused).squeeze(-1)
            return AttackOutput(
                logits=logits,
                embeddings=fused,
                auxiliary={"token_attention_weights": weights},
            )
else:
    class HybridAttack(TorchAttackModel):
        pass
