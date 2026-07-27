"""Which model-generated positions the teacher is asked to score.

Querying the teacher on every generated token is the simplest policy and the
most expensive one.  The selectors here trade coverage for cost under an
explicit budget.

Honest accounting
-----------------
Reducing the number of *selected output positions* reduces:

* the LM-head projection and softmax work on both sides,
* the teacher-target cache size and I/O,
* the number of student loss positions.

It does **not** proportionally reduce total teacher FLOPs: the teacher still
runs a full forward pass over the whole sequence to produce the hidden states.
Reports therefore label this ``teacher_queried_position_ratio``, never
"teacher compute saved".

Determinism
-----------
Sub-sampling is seeded from ``sha256(run_seed || trajectory_id)`` rather than
Python's salted :func:`hash`, so the same trajectory selects the same positions
in every process, on every OS, on every rerun.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field

from miniverl.config.models import SelectionConfig, SelectorName
from miniverl.errors import ConfigError
from miniverl.schemas.trajectory import Trajectory
from miniverl.trajectory.masks import (
    critical_target_positions,
    model_target_positions,
    positions_by_span_type,
)
from miniverl.utils.seeding import derive_seed

__all__ = [
    "SelectionStats",
    "SelectionResult",
    "select_positions",
    "aggregate_selection_stats",
    "derive_seed",
]


@dataclass(frozen=True)
class SelectionStats:
    """Accounting for one trajectory's teacher-query budget."""

    total_model_tokens: int
    selected_model_tokens: int
    total_critical_tokens: int
    selected_critical_tokens: int
    selector: str
    by_span_type: dict[str, int] = field(default_factory=dict)

    @property
    def query_ratio(self) -> float:
        """Fraction of model-generated tokens sent to the teacher."""
        if self.total_model_tokens == 0:
            return 0.0
        return self.selected_model_tokens / self.total_model_tokens

    def to_dict(self) -> dict[str, float | int | str | dict[str, int]]:
        """JSON-friendly view used by metrics and reports."""
        return {
            "selector": self.selector,
            "total_model_tokens": self.total_model_tokens,
            "selected_model_tokens": self.selected_model_tokens,
            "total_critical_tokens": self.total_critical_tokens,
            "selected_critical_tokens": self.selected_critical_tokens,
            "query_ratio": self.query_ratio,
            "by_span_type": dict(self.by_span_type),
        }


@dataclass(frozen=True)
class SelectionResult:
    """Selected target positions and their loss weights."""

    positions: list[int]
    weights: list[float]
    stats: SelectionStats

    def __len__(self) -> int:
        return len(self.positions)


def _apply_cap(positions: list[int], cap: int | None) -> list[int]:
    """Deterministically truncate to ``cap`` positions, keeping order."""
    if cap is None or len(positions) <= cap:
        return positions
    return positions[:cap]


def _deterministic_sample(candidates: Sequence[int], count: int, rng: random.Random) -> list[int]:
    """Sample ``count`` distinct entries and return them in ascending order."""
    if count <= 0:
        return []
    if count >= len(candidates):
        return list(candidates)
    return sorted(rng.sample(list(candidates), count))


def select_positions(
    trajectory: Trajectory,
    config: SelectionConfig,
    *,
    run_seed: int = 0,
) -> SelectionResult:
    """Choose target positions for teacher scoring, with weights.

    Returns *target* positions (the tokens being supervised).  The caller
    converts them to prediction positions via
    :func:`miniverl.trajectory.masks.prediction_positions`.
    """
    model_positions = model_target_positions(trajectory.model_generated_mask)
    critical_positions = critical_target_positions(
        trajectory.model_generated_mask, trajectory.critical_mask
    )
    critical_set = set(critical_positions)
    rng = random.Random(derive_seed(run_seed, trajectory.trajectory_id))

    selector = config.selector
    if selector is SelectorName.ALL_MODEL_TOKENS:
        chosen = list(model_positions)
    elif selector is SelectorName.TOOL_AND_FINAL:
        chosen = list(critical_positions)
    elif selector is SelectorName.UNIFORM_RATIO:
        budget = math.ceil(config.ratio * len(model_positions))
        chosen = _deterministic_sample(model_positions, budget, rng)
    elif selector is SelectorName.HYBRID:
        budget = math.ceil(config.ratio * len(model_positions))
        chosen = list(critical_positions)
        remaining = budget - len(chosen)
        if remaining > 0:
            others = [p for p in model_positions if p not in critical_set]
            chosen = sorted(chosen + _deterministic_sample(others, remaining, rng))
        else:
            chosen = sorted(chosen)
    else:  # pragma: no cover - exhaustive over the enum
        raise ConfigError(f"unknown selector {selector!r}")

    chosen = _apply_cap(sorted(chosen), config.max_positions_per_trajectory)
    weights = [config.critical_weight if p in critical_set else config.other_weight for p in chosen]

    stats = SelectionStats(
        total_model_tokens=len(model_positions),
        selected_model_tokens=len(chosen),
        total_critical_tokens=len(critical_positions),
        selected_critical_tokens=sum(1 for p in chosen if p in critical_set),
        selector=selector.value,
        by_span_type=positions_by_span_type(chosen, trajectory.spans),
    )
    return SelectionResult(positions=chosen, weights=weights, stats=stats)


def aggregate_selection_stats(stats: Sequence[SelectionStats]) -> dict[str, float | int | dict]:
    """Sum per-trajectory selection stats into run-level metrics."""
    total_model = sum(s.total_model_tokens for s in stats)
    selected = sum(s.selected_model_tokens for s in stats)
    by_span: dict[str, int] = {}
    for s in stats:
        for name, count in s.by_span_type.items():
            by_span[name] = by_span.get(name, 0) + count
    return {
        "trajectories": len(stats),
        "total_model_tokens": total_model,
        "selected_model_tokens": selected,
        "teacher_queried_position_ratio": (selected / total_model) if total_model else 0.0,
        "total_critical_tokens": sum(s.total_critical_tokens for s in stats),
        "selected_critical_tokens": sum(s.selected_critical_tokens for s in stats),
        "selected_by_span_type": by_span,
    }
