"""Mixture-of-Experts attack model."""

from __future__ import annotations

from .base import AttackBatch, AttackOutput
from .torch_base import TorchAttackModel, nn, torch


if nn is not None:
    class _Expert(nn.Module):
        def __init__(self, input_dim, hidden_dim, dropout):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)


    class MixtureOfExpertsAttack(TorchAttackModel):
        DEFAULT_EXPERTS = (
            "loss",
            "entropy",
            "residual",
            "neighbourhood",
            "rank",
        )

        def __init__(self, config) -> None:
            super().__init__(config)
            names = tuple(
                config.model_kwargs.get(
                    "expert_names", self.DEFAULT_EXPERTS
                )
            )
            hidden = config.model_kwargs.get("expert_hidden_dim", 64)
            gate_hidden = config.model_kwargs.get("gate_hidden_dim", 64)
            self.expert_names = names
            self.experts = nn.ModuleDict({
                name: _Expert(
                    config.global_feature_dim, hidden, config.dropout
                )
                for name in names
            })
            self.gate = nn.Sequential(
                nn.Linear(config.global_feature_dim, gate_hidden),
                nn.GELU(),
                nn.Linear(gate_hidden, len(names)),
            )
            # TODO: route separate feature groups to each expert.

        def forward(self, batch: AttackBatch) -> AttackOutput:
            if batch.global_features is None:
                raise ValueError("global_features are required.")
            expert_logits = torch.stack([
                self.experts[name](batch.global_features)
                for name in self.expert_names
            ], dim=-1)
            gate_weights = torch.softmax(
                self.gate(batch.global_features), dim=-1
            )
            logits = (expert_logits * gate_weights).sum(dim=-1)
            return AttackOutput(
                logits=logits,
                auxiliary={
                    "expert_logits": expert_logits,
                    "gate_weights": gate_weights,
                    "expert_names": self.expert_names,
                },
            )
else:
    class MixtureOfExpertsAttack(TorchAttackModel):
        pass
