"""Provenance masks and the causal target/prediction position convention.

The single most dangerous bug in a distillation trainer is an off-by-one
between the token being predicted and the distribution used to predict it.
miniVERL therefore keeps two vocabularies of the word "position":

*target position* ``j``
    The index of the token whose identity is being supervised.

*prediction position* ``j - 1``
    The index whose output distribution predicts the token at ``j``.

Nothing in the codebase converts between them implicitly.  Every conversion
goes through :func:`prediction_positions`, and ``j = 0`` is always rejected
because no distribution precedes it.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from miniverl.errors import TrajectoryError
from miniverl.schemas.trajectory import Span

__all__ = [
    "build_masks",
    "model_target_positions",
    "critical_target_positions",
    "prediction_positions",
    "validate_target_positions",
    "positions_by_span_type",
]


def build_masks(spans: Sequence[Span], length: int) -> tuple[list[bool], list[bool]]:
    """Derive ``(model_generated_mask, critical_mask)`` from a span partition."""
    model = [False] * length
    critical = [False] * length
    for span in spans:
        if span.end > length:
            raise TrajectoryError(
                f"span {span.span_type.value} ends at {span.end} but the sequence "
                f"has only {length} tokens"
            )
        if span.is_model_generated:
            for i in range(span.start, span.end):
                model[i] = True
        if span.is_critical:
            for i in range(span.start, span.end):
                critical[i] = True
    return model, critical


def model_target_positions(model_generated_mask: Sequence[bool]) -> list[int]:
    """Target positions that may be supervised.

    Position ``0`` is excluded even when marked model-generated: there is no
    preceding distribution that could predict it.
    """
    return [j for j, flag in enumerate(model_generated_mask) if flag and j > 0]


def critical_target_positions(
    model_generated_mask: Sequence[bool], critical_mask: Sequence[bool]
) -> list[int]:
    """Target positions inside tool-call or final-answer spans."""
    return [
        j
        for j, (m, c) in enumerate(zip(model_generated_mask, critical_mask, strict=False))
        if m and c and j > 0
    ]


def prediction_positions(target_positions: Iterable[int]) -> list[int]:
    """Map target positions to the prediction positions that produce them."""
    out: list[int] = []
    for j in target_positions:
        if j <= 0:
            raise TrajectoryError(
                f"target position {j} has no preceding prediction position; "
                "position 0 can never be a training target"
            )
        out.append(j - 1)
    return out


def validate_target_positions(
    target_positions: Sequence[int],
    model_generated_mask: Sequence[bool],
) -> None:
    """Assert that every target position is a model-generated token.

    This is the guard that makes "tool outputs are context, not labels" a
    checked property rather than a comment.
    """
    n = len(model_generated_mask)
    seen: set[int] = set()
    previous = -1
    for j in target_positions:
        if not 0 <= j < n:
            raise TrajectoryError(f"target position {j} is outside [0, {n})")
        if j == 0:
            raise TrajectoryError("position 0 can never be a training target")
        if not model_generated_mask[j]:
            raise TrajectoryError(
                f"target position {j} is not a model-generated token; system, user "
                "and tool-result tokens must never be supervised"
            )
        if j in seen:
            raise TrajectoryError(f"target position {j} appears more than once")
        if j <= previous:
            raise TrajectoryError("target positions must be strictly increasing")
        seen.add(j)
        previous = j


def positions_by_span_type(positions: Sequence[int], spans: Sequence[Span]) -> dict[str, int]:
    """Count selected positions per span type."""
    counts: dict[str, int] = {}
    span_iter = list(spans)
    idx = 0
    for pos in positions:
        while idx < len(span_iter) and pos >= span_iter[idx].end:
            idx += 1
        if idx >= len(span_iter):
            raise TrajectoryError(f"position {pos} is outside the span partition")
        name = span_iter[idx].span_type.value
        counts[name] = counts.get(name, 0) + 1
    return counts
