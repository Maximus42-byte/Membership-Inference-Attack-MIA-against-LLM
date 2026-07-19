"""Attack-model architecture scaffolds for ERN-MIA."""

from .base import AttackBatch, AttackModelConfig, AttackOutput
from .registry import ATTACK_MODEL_REGISTRY, create_attack_model

__all__ = [
    "AttackBatch",
    "AttackModelConfig",
    "AttackOutput",
    "ATTACK_MODEL_REGISTRY",
    "create_attack_model",
]
