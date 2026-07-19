"""Central attack-model registry."""

from __future__ import annotations

from .base import AttackModelConfig, BaseAttackModel
from .sklearn_models import (
    CatBoostAttack,
    GradientBoostingAttack,
    LightGBMAttack,
    LinearSVMAttack,
    LogisticRegressionAttack,
    RandomForestAttack,
    RBFSVMAttack,
    XGBoostAttack,
)
from .mlp import MLPAttack
from .token_cnn import TokenCNNAttack
from .recurrent import BiLSTMAttack, GRUAttack
from .transformer import TransformerAttack
from .hybrid import HybridAttack
from .mixture_of_experts import MixtureOfExpertsAttack


ATTACK_MODEL_REGISTRY = {
    "logistic_regression": LogisticRegressionAttack,
    "linear_svm": LinearSVMAttack,
    "rbf_svm": RBFSVMAttack,
    "random_forest": RandomForestAttack,
    "gradient_boosting": GradientBoostingAttack,
    "xgboost": XGBoostAttack,
    "lightgbm": LightGBMAttack,
    "catboost": CatBoostAttack,
    "mlp": MLPAttack,
    "token_cnn": TokenCNNAttack,
    "bilstm": BiLSTMAttack,
    "gru": GRUAttack,
    "transformer": TransformerAttack,
    "hybrid": HybridAttack,
    "mixture_of_experts": MixtureOfExpertsAttack,
}


def create_attack_model(config: AttackModelConfig) -> BaseAttackModel:
    config.validate()
    try:
        model_cls = ATTACK_MODEL_REGISTRY[config.model_type]
    except KeyError as exc:
        available = ", ".join(sorted(ATTACK_MODEL_REGISTRY))
        raise ValueError(
            f"Unknown model_type={config.model_type!r}. "
            f"Available: {available}"
        ) from exc
    return model_cls(config)
