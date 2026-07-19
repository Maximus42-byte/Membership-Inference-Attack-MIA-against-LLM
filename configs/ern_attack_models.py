"""Draft configurations for ERN-MIA attack architectures."""

from attack_models import AttackModelConfig


LOGISTIC = AttackModelConfig(
    model_type="logistic_regression",
    global_feature_dim=20,
    class_weight="balanced",
    model_kwargs={"C": 1.0, "max_iter": 1000},
)

XGBOOST = AttackModelConfig(
    model_type="xgboost",
    global_feature_dim=20,
    model_kwargs={
        "n_estimators": 500,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
)

MLP = AttackModelConfig(
    model_type="mlp",
    global_feature_dim=20,
    hidden_dims=(128, 64),
    dropout=0.2,
)

TOKEN_CNN = AttackModelConfig(
    model_type="token_cnn",
    token_feature_dim=6,
    dropout=0.2,
    model_kwargs={
        "channels": 128,
        "kernel_sizes": (3, 5, 7),
    },
)

BILSTM = AttackModelConfig(
    model_type="bilstm",
    token_feature_dim=6,
    dropout=0.2,
    model_kwargs={
        "hidden_size": 128,
        "num_layers": 2,
    },
)

TRANSFORMER = AttackModelConfig(
    model_type="transformer",
    token_feature_dim=6,
    dropout=0.2,
    model_kwargs={
        "model_dim": 128,
        "num_heads": 4,
        "num_layers": 3,
        "feedforward_dim": 256,
    },
)

HYBRID = AttackModelConfig(
    model_type="hybrid",
    global_feature_dim=20,
    token_feature_dim=6,
    dropout=0.2,
)

MIXTURE_OF_EXPERTS = AttackModelConfig(
    model_type="mixture_of_experts",
    global_feature_dim=20,
    dropout=0.2,
)
