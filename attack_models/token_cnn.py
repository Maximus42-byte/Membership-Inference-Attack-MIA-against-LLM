"""Token-level CNN for local memorization patterns."""

from __future__ import annotations

from .base import AttackBatch, AttackOutput
from .torch_base import TorchAttackModel, nn, torch


if nn is not None:
    class TokenCNNAttack(TorchAttackModel):
        def __init__(self, config) -> None:
            super().__init__(config)
            channels = config.model_kwargs.get("channels", 128)
            kernels = tuple(config.model_kwargs.get("kernel_sizes", (3, 5, 7)))
            self.branches = nn.ModuleList([
                nn.Sequential(
                    nn.Conv1d(
                        config.token_feature_dim,
                        channels,
                        kernel_size=k,
                        padding=k // 2,
                    ),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                )
                for k in kernels
            ])
            self.classifier = nn.Linear(channels * len(kernels), 1)

        def forward(self, batch: AttackBatch) -> AttackOutput:
            if batch.token_features is None:
                raise ValueError("token_features are required.")
            x = batch.token_features.transpose(1, 2)
            pooled_parts = []
            for branch in self.branches:
                hidden = branch(x)
                if batch.attention_mask is not None:
                    mask = batch.attention_mask.unsqueeze(1).bool()
                    hidden = hidden.masked_fill(~mask, float("-inf"))
                pooled_parts.append(hidden.amax(dim=-1))
            embedding = torch.cat(pooled_parts, dim=-1)
            logits = self.classifier(embedding).squeeze(-1)
            return AttackOutput(logits=logits, embeddings=embedding)
else:
    class TokenCNNAttack(TorchAttackModel):
        pass
