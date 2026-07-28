"""Diagnostic policy-evaluation metrics that do not replace strict scoring."""

from __future__ import annotations

import re

from miniverl.schemas.trajectory import Trajectory

__all__ = [
    "TOLERANCE",
    "lenient_answer_matches",
    "lenient_diagnostic_success_rate",
    "unwrap_answer",
]

TOLERANCE = 1.0e-3
_ANSWER_BLOCK = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)


def unwrap_answer(predicted: str) -> str:
    """Strip a teacher-style answer wrapper, if one is present."""
    match = _ANSWER_BLOCK.search(predicted)
    return (match.group(1) if match else predicted).strip()


def _as_number(text: str) -> float | None:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def lenient_answer_matches(predicted: str, expected: str) -> bool:
    """Compare answers after unwrapping, with a small numeric tolerance."""
    unwrapped = unwrap_answer(predicted)
    got, want = _as_number(unwrapped), _as_number(expected)
    if got is None or want is None:
        return unwrapped == expected.strip()
    return abs(got - want) <= TOLERANCE * max(1.0, abs(want))


def lenient_diagnostic_success_rate(trajectories: list[Trajectory]) -> float:
    """Return strict successes plus presentation-only failures over all tasks."""
    if not trajectories:
        return 0.0
    solved = 0
    for trajectory in trajectories:
        verification = trajectory.verification
        if verification is None:
            continue
        if verification.solved or lenient_answer_matches(
            verification.predicted or "",
            verification.expected or "",
        ):
            solved += 1
    return solved / len(trajectories)
