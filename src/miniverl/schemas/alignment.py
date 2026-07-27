"""Explicit student/teacher position alignment.

Causal convention used everywhere in miniVERL
---------------------------------------------
For a target token sitting at absolute index ``j`` of a sequence, the model
distribution that predicts it lives at index ``j - 1``.  A *prediction
position* is therefore always ``j - 1``, and ``j = 0`` can never be a target
because nothing predicts it.

Under ``teacher.mode = "standard"`` the teacher sees the byte-identical
context, so ``teacher_prediction_positions == student_prediction_positions``.
Under ``teacher.mode = "privileged_context"`` the teacher sequence carries an
extra oracle block, so the positions differ by a per-span offset and must be
carried around explicitly -- never assumed equal.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator

__all__ = ["AlignmentMap"]


class AlignmentMap(BaseModel):
    """Aligned prediction positions for one trajectory.

    All five parallel lists have the same length ``N`` = number of selected
    target tokens.  Entry ``i`` says: *the student distribution at
    ``student_prediction_positions[i]`` and the teacher distribution at
    ``teacher_prediction_positions[i]`` both predict token
    ``target_token_ids[i]``, and it contributes with weight
    ``token_weights[i]``.*
    """

    model_config = ConfigDict(extra="forbid")

    trajectory_id: str
    student_prediction_positions: list[int]
    teacher_prediction_positions: list[int]
    target_token_ids: list[int]
    model_token_mask: list[bool]
    token_weights: list[float]
    span_types: list[str]

    @model_validator(mode="after")
    def _validate(self) -> AlignmentMap:
        n = len(self.student_prediction_positions)
        fields: dict[str, Sequence[Any]] = {
            "teacher_prediction_positions": self.teacher_prediction_positions,
            "target_token_ids": self.target_token_ids,
            "model_token_mask": self.model_token_mask,
            "token_weights": self.token_weights,
            "span_types": self.span_types,
        }
        for name, seq in fields.items():
            if len(seq) != n:
                raise ValueError(f"alignment field '{name}' has length {len(seq)}, expected {n}")
        if n == 0:
            return self
        if any(p < 0 for p in self.student_prediction_positions):
            raise ValueError("student prediction positions must be >= 0")
        if any(p < 0 for p in self.teacher_prediction_positions):
            raise ValueError("teacher prediction positions must be >= 0")
        strictly_increasing = all(
            b > a
            for a, b in zip(
                self.student_prediction_positions,
                self.student_prediction_positions[1:],
                strict=False,
            )
        )
        if not strictly_increasing:
            raise ValueError("student prediction positions must be strictly increasing")
        if any(w < 0.0 for w in self.token_weights):
            raise ValueError("token weights must be non-negative")
        for i, in_mask in enumerate(self.model_token_mask):
            if not in_mask and self.token_weights[i] != 0.0:
                raise ValueError(
                    "a non-model token was given a non-zero weight; tool/user/system "
                    "tokens must never contribute to the distillation loss"
                )
        return self

    @property
    def num_positions(self) -> int:
        """Number of aligned target tokens."""
        return len(self.student_prediction_positions)

    @property
    def total_weight(self) -> float:
        """Sum of token weights, i.e. the loss normalizer."""
        return float(sum(self.token_weights))

    def is_identity(self) -> bool:
        """``True`` when teacher and student positions coincide exactly."""
        return self.student_prediction_positions == self.teacher_prediction_positions

    def counts_by_span_type(self) -> dict[str, int]:
        """Selected-token counts keyed by span type."""
        counts: dict[str, int] = {}
        for span_type in self.span_types:
            counts[span_type] = counts.get(span_type, 0) + 1
        return counts
