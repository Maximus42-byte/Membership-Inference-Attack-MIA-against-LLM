"""ERN-MIA attack-vector extraction scaffold.

This module is intentionally not connected to the current execution pipeline.
Later it should consume outputs produced by run_mia_unified.py and the existing
Neighbourhood, Residual, and RRN attack implementations.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Mapping, Sequence


@dataclass(slots=True)
class ERNGlobalFeatures:
    # Base target-model features
    mean_log_likelihood: float | None = None
    token_loss_std: float | None = None
    min_k_20_score: float | None = None
    sequence_length: int | None = None

    # Entropy features
    entropy_mean: float | None = None
    entropy_std: float | None = None
    entropy_min: float | None = None

    # Reference-model features
    reference_log_likelihood: float | None = None
    residual_log_likelihood: float | None = None
    residual_entropy: float | None = None

    # Neighbourhood features
    neighbourhood_score: float | None = None
    z_neighbourhood_score: float | None = None
    rn_score: float | None = None

    # Rank features
    q_score: float | None = None
    q_p_value: float | None = None
    rrn_score: float | None = None
    rrn_p_value: float | None = None

    # Neighbour quality
    neighbour_similarity_mean: float | None = None
    neighbour_similarity_std: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ERNTokenFeatures:
    """Token feature sequence.

    Intended per-token channels:
        target_loss
        target_entropy
        reference_loss
        reference_entropy
        residual_loss
        residual_entropy
    """

    values: Sequence[Sequence[float]]
    attention_mask: Sequence[int]


class ERNFeatureExtractor:
    """Future adapter around the existing repository functions."""

    def __init__(
        self,
        *,
        target_model: Any,
        target_tokenizer: Any,
        reference_model: Any | None = None,
        reference_tokenizer: Any | None = None,
    ) -> None:
        self.target_model = target_model
        self.target_tokenizer = target_tokenizer
        self.reference_model = reference_model
        self.reference_tokenizer = reference_tokenizer

    def extract_global_features(
        self,
        text: str,
        *,
        neighbours: Sequence[str] | None = None,
        existing_scores: Mapping[str, Any] | None = None,
    ) -> ERNGlobalFeatures:
        """Extract or merge one global Attack Vector.

        TODO:
            1. Reuse likelihood helpers from run_mia_unified.py.
            2. Reuse neighbourhood score from calibration_attack.py.
            3. Reuse residual calculations from
               Residual_neighborhood_attack.py.
            4. Reuse RRN score from RRN_neighborhood_attack.py.
            5. Add entropy and Min-k% features.
            6. Define score direction and missing-value policy.
        """
        raise NotImplementedError(
            "Feature extraction must be connected to current repository code."
        )

    def extract_token_features(self, text: str) -> ERNTokenFeatures:
        """Build token-level loss, entropy, and residual sequences."""
        raise NotImplementedError(
            "Token-level feature extraction is not implemented."
        )

    def extract(
        self,
        text: str,
        *,
        neighbours: Sequence[str] | None = None,
        existing_scores: Mapping[str, Any] | None = None,
        include_token_features: bool = False,
    ) -> dict[str, Any]:
        global_features = self.extract_global_features(
            text,
            neighbours=neighbours,
            existing_scores=existing_scores,
        )
        output: dict[str, Any] = {
            "global_features": global_features.to_dict(),
        }
        if include_token_features:
            token_features = self.extract_token_features(text)
            output["token_features"] = token_features.values
            output["attention_mask"] = token_features.attention_mask
        return output
