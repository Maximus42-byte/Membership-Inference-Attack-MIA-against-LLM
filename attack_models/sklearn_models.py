"""Tabular attack-model options.

Optional dependencies are imported only when the corresponding model is created.
"""

from __future__ import annotations

from typing import Any

from .base import AttackModelConfig, BaseAttackModel


class SklearnAttackModel(BaseAttackModel):
    estimator: Any = None

    def fit(self, features: Any, labels: Any, **kwargs: Any) -> "SklearnAttackModel":
        if self.estimator is None:
            raise RuntimeError("Estimator was not initialized.")
        self.estimator.fit(features, labels, **kwargs)
        self.is_fitted = True
        return self

    def predict_score(self, features: Any, **kwargs: Any) -> Any:
        self.require_fitted()
        if hasattr(self.estimator, "predict_proba"):
            return self.estimator.predict_proba(features, **kwargs)[:, 1]
        if hasattr(self.estimator, "decision_function"):
            return self.estimator.decision_function(features, **kwargs)
        raise TypeError("Estimator has no member-oriented continuous output.")

    def predict_proba(self, features: Any, **kwargs: Any) -> Any:
        self.require_fitted()
        if not hasattr(self.estimator, "predict_proba"):
            raise NotImplementedError(
                "This estimator requires an explicit calibration wrapper."
            )
        return self.estimator.predict_proba(features, **kwargs)


class LogisticRegressionAttack(SklearnAttackModel):
    def __init__(self, config: AttackModelConfig) -> None:
        super().__init__(config)
        try:
            from sklearn.linear_model import LogisticRegression
        except ImportError as exc:
            raise ImportError("Install scikit-learn.") from exc

        self.estimator = LogisticRegression(
            C=config.model_kwargs.get("C", 1.0),
            penalty=config.model_kwargs.get("penalty", "l2"),
            solver=config.model_kwargs.get("solver", "lbfgs"),
            max_iter=config.model_kwargs.get("max_iter", 1000),
            class_weight=config.class_weight,
            random_state=config.random_state,
        )
        # TODO: preprocessing pipeline and coefficient report.


class LinearSVMAttack(SklearnAttackModel):
    def __init__(self, config: AttackModelConfig) -> None:
        super().__init__(config)
        try:
            from sklearn.svm import LinearSVC
        except ImportError as exc:
            raise ImportError("Install scikit-learn.") from exc

        self.estimator = LinearSVC(
            C=config.model_kwargs.get("C", 1.0),
            class_weight=config.class_weight,
            random_state=config.random_state,
        )
        # TODO: CalibratedClassifierCV with a dedicated calibration split.


class RBFSVMAttack(SklearnAttackModel):
    def __init__(self, config: AttackModelConfig) -> None:
        super().__init__(config)
        try:
            from sklearn.svm import SVC
        except ImportError as exc:
            raise ImportError("Install scikit-learn.") from exc

        self.estimator = SVC(
            C=config.model_kwargs.get("C", 1.0),
            gamma=config.model_kwargs.get("gamma", "scale"),
            kernel="rbf",
            probability=False,
            class_weight=config.class_weight,
            random_state=config.random_state,
        )
        # TODO: external probability calibration.


class RandomForestAttack(SklearnAttackModel):
    def __init__(self, config: AttackModelConfig) -> None:
        super().__init__(config)
        try:
            from sklearn.ensemble import RandomForestClassifier
        except ImportError as exc:
            raise ImportError("Install scikit-learn.") from exc

        self.estimator = RandomForestClassifier(
            n_estimators=config.model_kwargs.get("n_estimators", 500),
            max_depth=config.model_kwargs.get("max_depth"),
            min_samples_leaf=config.model_kwargs.get("min_samples_leaf", 2),
            max_features=config.model_kwargs.get("max_features", "sqrt"),
            class_weight=config.class_weight,
            random_state=config.random_state,
            n_jobs=config.model_kwargs.get("n_jobs", -1),
        )


class GradientBoostingAttack(SklearnAttackModel):
    def __init__(self, config: AttackModelConfig) -> None:
        super().__init__(config)
        try:
            from sklearn.ensemble import HistGradientBoostingClassifier
        except ImportError as exc:
            raise ImportError("Install scikit-learn.") from exc

        self.estimator = HistGradientBoostingClassifier(
            learning_rate=config.model_kwargs.get("learning_rate", 0.05),
            max_iter=config.model_kwargs.get("max_iter", 300),
            max_leaf_nodes=config.model_kwargs.get("max_leaf_nodes", 31),
            l2_regularization=config.model_kwargs.get("l2_regularization", 1.0),
            random_state=config.random_state,
        )


class XGBoostAttack(SklearnAttackModel):
    def __init__(self, config: AttackModelConfig) -> None:
        super().__init__(config)
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError("Install xgboost.") from exc

        self.estimator = XGBClassifier(
            n_estimators=config.model_kwargs.get("n_estimators", 500),
            max_depth=config.model_kwargs.get("max_depth", 5),
            learning_rate=config.model_kwargs.get("learning_rate", 0.05),
            subsample=config.model_kwargs.get("subsample", 0.8),
            colsample_bytree=config.model_kwargs.get("colsample_bytree", 0.8),
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=config.random_state,
        )
        # TODO: early stopping, scale_pos_weight, SHAP.


class LightGBMAttack(SklearnAttackModel):
    def __init__(self, config: AttackModelConfig) -> None:
        super().__init__(config)
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise ImportError("Install lightgbm.") from exc

        self.estimator = LGBMClassifier(
            n_estimators=config.model_kwargs.get("n_estimators", 500),
            num_leaves=config.model_kwargs.get("num_leaves", 31),
            learning_rate=config.model_kwargs.get("learning_rate", 0.05),
            class_weight=config.class_weight,
            random_state=config.random_state,
        )


class CatBoostAttack(SklearnAttackModel):
    def __init__(self, config: AttackModelConfig) -> None:
        super().__init__(config)
        try:
            from catboost import CatBoostClassifier
        except ImportError as exc:
            raise ImportError("Install catboost.") from exc

        self.estimator = CatBoostClassifier(
            iterations=config.model_kwargs.get("iterations", 500),
            depth=config.model_kwargs.get("depth", 6),
            learning_rate=config.model_kwargs.get("learning_rate", 0.05),
            loss_function="Logloss",
            verbose=False,
            random_seed=config.random_state,
        )
