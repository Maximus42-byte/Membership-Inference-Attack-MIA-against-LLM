"""MLP attack model for fixed-dimensional ERN vectors."""

from __future__ import annotations

from .base import AttackBatch, AttackOutput
from .torch_base import TorchAttackModel, nn


if nn is not None:
    class MLPAttack(TorchAttackModel):
        def __init__(self, config) -> None:
            super().__init__(config)
            if config.global_feature_dim <= 0:
                raise ValueError("global_feature_dim must be positive.")

            dims = (config.global_feature_dim, *config.hidden_dims, 1)
            layers = []
            for in_dim, out_dim in zip(dims[:-2], dims[1:-1]):
                layers.extend([
                    nn.Linear(in_dim, out_dim),
                    nn.LayerNorm(out_dim),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                ])
            layers.append(nn.Linear(dims[-2], 1))
            self.network = nn.Sequential(*layers)

        def forward(self, batch: AttackBatch) -> AttackOutput:
            if batch.global_features is None:
                raise ValueError("global_features are required.")
            logits = self.network(batch.global_features).squeeze(-1)
            return AttackOutput(logits=logits)
else:
    class MLPAttack(TorchAttackModel):
        pass
