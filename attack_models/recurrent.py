"""BiLSTM and GRU attack models."""

from __future__ import annotations

from .base import AttackBatch, AttackOutput
from .torch_base import TorchAttackModel, nn, torch


if nn is not None:
    class RecurrentAttack(TorchAttackModel):
        recurrent_type = "lstm"

        def __init__(self, config) -> None:
            super().__init__(config)
            hidden_size = config.model_kwargs.get("hidden_size", 128)
            num_layers = config.model_kwargs.get("num_layers", 2)
            recurrent_cls = nn.LSTM if self.recurrent_type == "lstm" else nn.GRU
            self.encoder = recurrent_cls(
                input_size=config.token_feature_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=config.dropout if num_layers > 1 else 0.0,
                bidirectional=True,
            )
            self.attention = nn.Linear(hidden_size * 2, 1)
            self.classifier = nn.Linear(hidden_size * 2, 1)

        def forward(self, batch: AttackBatch) -> AttackOutput:
            if batch.token_features is None:
                raise ValueError("token_features are required.")
            hidden, _ = self.encoder(batch.token_features)
            attn_logits = self.attention(hidden).squeeze(-1)
            if batch.attention_mask is not None:
                attn_logits = attn_logits.masked_fill(
                    ~batch.attention_mask.bool(), float("-inf")
                )
            weights = torch.softmax(attn_logits, dim=-1)
            embedding = (hidden * weights.unsqueeze(-1)).sum(dim=1)
            logits = self.classifier(embedding).squeeze(-1)
            return AttackOutput(
                logits=logits,
                embeddings=embedding,
                auxiliary={"attention_weights": weights},
            )

    class BiLSTMAttack(RecurrentAttack):
        recurrent_type = "lstm"

    class GRUAttack(RecurrentAttack):
        recurrent_type = "gru"
else:
    class RecurrentAttack(TorchAttackModel):
        pass
    class BiLSTMAttack(RecurrentAttack):
        pass
    class GRUAttack(RecurrentAttack):
        pass
