"""PyTorch base class for neural attack models."""

from __future__ import annotations

from typing import Any

from .base import AttackBatch, AttackModelConfig, AttackOutput, BaseAttackModel

try:
    import torch
    from torch import nn
except ImportError:
    torch = None
    nn = None


if nn is not None:
    class TorchAttackModel(nn.Module, BaseAttackModel):
        def __init__(self, config: AttackModelConfig) -> None:
            nn.Module.__init__(self)
            BaseAttackModel.__init__(self, config)

        def fit(self, features: Any, labels: Any, **kwargs: Any):
            raise NotImplementedError(
                "Neural trainer will be implemented in ERN_attack_model.py."
            )

        def predict_score(self, features: Any, **kwargs: Any):
            raise NotImplementedError(
                "Device-aware batched inference is not implemented."
            )

        def compute_loss(self, output: AttackOutput, labels: Any):
            labels = labels.float().view_as(output.logits)
            return nn.functional.binary_cross_entropy_with_logits(
                output.logits, labels
            )
else:
    class TorchAttackModel(BaseAttackModel):
        def __init__(self, config: AttackModelConfig) -> None:
            super().__init__(config)
            raise ImportError("Neural attack models require PyTorch.")

        def fit(self, features: Any, labels: Any, **kwargs: Any):
            raise NotImplementedError

        def predict_score(self, features: Any, **kwargs: Any):
            raise NotImplementedError
