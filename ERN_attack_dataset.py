"""Dataset representation for ERN-MIA Attack Model training."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(slots=True)
class ERNAttackExample:
    sample_id: str
    membership_label: int
    global_features: Mapping[str, float | int | None]
    token_features: Sequence[Sequence[float]] | None = None
    attention_mask: Sequence[int] | None = None
    model_id: str | None = None
    split_name: str | None = None

    def validate(self) -> None:
        if self.membership_label not in (0, 1):
            raise ValueError("membership_label must be 0 or 1.")
        if not self.sample_id:
            raise ValueError("sample_id cannot be empty.")


def save_examples_jsonl(
    examples: Sequence[ERNAttackExample],
    output_path: str | Path,
) -> None:
    """Simple interim storage format.

    TODO:
        For large token arrays, replace JSONL with NPZ, Arrow, or Parquet.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for example in examples:
            example.validate()
            stream.write(json.dumps(asdict(example), ensure_ascii=False) + "\n")


def load_examples_jsonl(
    input_path: str | Path,
) -> list[ERNAttackExample]:
    examples = []
    with Path(input_path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                example = ERNAttackExample(**json.loads(line))
                example.validate()
            except Exception as exc:
                raise ValueError(
                    f"Invalid example at line {line_number}."
                ) from exc
            examples.append(example)
    return examples


def split_by_model_id(*args: Any, **kwargs: Any):
    """Placeholder for shadow-model-aware splitting."""
    raise NotImplementedError(
        "Cross-model and shadow-model splitting is not implemented."
    )
