"""Transformer over token-level loss, entropy, and residual features."""

from __future__ import annotations

from .base import AttackBatch, AttackOutput
from .torch_base import TorchAttackModel, nn, torch


if nn is not None:
    class TransformerAttack(TorchAttackModel):
        def __init__(self, config) -> None:
            super().__init__(config)
            model_dim = config.model_kwargs.get("model_dim", 128)
            nhead = config.model_kwargs.get("num_heads", 4)
            num_layers = config.model_kwargs.get("num_layers", 3)
            ff_dim = config.model_kwargs.get("feedforward_dim", 256)

            self.input_projection = nn.Linear(
                config.token_feature_dim, model_dim
            )
            layer = nn.TransformerEncoderLayer(
                d_model=model_dim,
                nhead=nhead,
                dim_feedforward=ff_dim,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
            self.pool_query = nn.Parameter(torch.randn(model_dim))
            self.classifier = nn.Linear(model_dim, 1)
            # TODO: positional encoding and long-sequence policy.

        def forward(self, batch: AttackBatch) -> AttackOutput:
            if batch.token_features is None:
                raise ValueError("token_features are required.")
            hidden = self.input_projection(batch.token_features)
            padding_mask = None
            if batch.attention_mask is not None:
                padding_mask = ~batch.attention_mask.bool()
            hidden = self.encoder(
                hidden,
                src_key_padding_mask=padding_mask,
            )
            attn_logits = torch.einsum("btd,d->bt", hidden, self.pool_query)
            if padding_mask is not None:
                attn_logits = attn_logits.masked_fill(
                    padding_mask, float("-inf")
                )
            weights = torch.softmax(attn_logits, dim=-1)
            embedding = (hidden * weights.unsqueeze(-1)).sum(dim=1)
            logits = self.classifier(embedding).squeeze(-1)
            return AttackOutput(
                logits=logits,
                embeddings=embedding,
                auxiliary={"attention_weights": weights},
            )
else:
    class TransformerAttack(TorchAttackModel):
        pass
