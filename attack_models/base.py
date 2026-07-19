"""Shared interfaces for all ERN-MIA attack models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(slots=True)
class AttackModelConfig:
    model_type: str
    global_feature_dim: int = 0
    token_feature_dim: int = 0
    hidden_dims: tuple[int, ...] = (128, 64)
    dropout: float = 0.2
    activation: str = "gelu"
    random_state: int = 42
    class_weight: str | dict[int, float] | None = None
    model_kwargs: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.model_type:
            raise ValueError("model_type cannot be empty.")
        if self.global_feature_dim < 0 or self.token_feature_dim < 0:
            raise ValueError("Feature dimensions cannot be negative.")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1).")


@dataclass(slots=True)
class AttackBatch:
    global_features: Any | None = None
    token_features: Any | None = None
    attention_mask: Any | None = None
    labels: Any | None = None
    metadata: Mapping[str, Any] | None = None


@dataclass(slots=True)
class AttackOutput:
    logits: Any
    probabilities: Any | None = None
    embeddings: Any | None = None
    auxiliary: Mapping[str, Any] | None = None


class BaseAttackModel(ABC):
    """Framework-neutral interface.

    Score direction convention:
        larger score = more member-like
    """

    def __init__(self, config: AttackModelConfig) -> None:
        config.validate()
        self.config = config
        self.is_fitted = False

    @abstractmethod
    def fit(self, features: Any, labels: Any, **kwargs: Any) -> "BaseAttackModel":
        raise NotImplementedError

    @abstractmethod
    def predict_score(self, features: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def predict_proba(self, features: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        raise NotImplementedError("Model serialization is not finalized.")

    @classmethod
    def load(cls, path: str | Path) -> "BaseAttackModel":
        raise NotImplementedError("Model loading is not finalized.")

    def require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("Attack model must be fitted before prediction.")
