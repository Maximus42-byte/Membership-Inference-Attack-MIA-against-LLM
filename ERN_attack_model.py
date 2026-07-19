"""Future entry point for ERN-MIA Attack Model experiments.

The current file defines argument structure and architectural intent only.
It deliberately stops before loading data or training a model.
"""

from __future__ import annotations

import argparse

from attack_models import AttackModelConfig, create_attack_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ERN-MIA Attack Model scaffold"
    )
    parser.add_argument(
        "--attack_features",
        type=str,
        required=True,
        help="Future JSONL/NPZ attack-feature dataset.",
    )
    parser.add_argument(
        "--attack_model",
        type=str,
        default="logistic_regression",
        choices=[
            "logistic_regression",
            "linear_svm",
            "rbf_svm",
            "random_forest",
            "gradient_boosting",
            "xgboost",
            "lightgbm",
            "catboost",
            "mlp",
            "token_cnn",
            "bilstm",
            "gru",
            "transformer",
            "hybrid",
            "mixture_of_experts",
        ],
    )
    parser.add_argument("--global_feature_dim", type=int, default=0)
    parser.add_argument("--token_feature_dim", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="results/ern_mia")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def build_model_from_args(args: argparse.Namespace):
    config = AttackModelConfig(
        model_type=args.attack_model,
        global_feature_dim=args.global_feature_dim,
        token_feature_dim=args.token_feature_dim,
        random_state=args.seed,
    )
    return create_attack_model(config)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    _ = build_model_from_args(args)

    raise NotImplementedError(
        "ERN-MIA training is intentionally disabled in this scaffold. "
        "Complete feature loading, preprocessing, train/validation/"
        "calibration/test splitting, training, and Low-FPR evaluation first."
    )


if __name__ == "__main__":
    main()
