"""Trajectory construction, provenance masks, alignment and serialization."""

from __future__ import annotations

from miniverl.trajectory.alignment import build_alignment_map, identity_alignment
from miniverl.trajectory.io import (
    iter_trajectories,
    read_trajectories,
    write_trajectories,
)
from miniverl.trajectory.masks import (
    build_masks,
    critical_target_positions,
    model_target_positions,
    prediction_positions,
    validate_target_positions,
)

__all__ = [
    "build_alignment_map",
    "identity_alignment",
    "iter_trajectories",
    "read_trajectories",
    "write_trajectories",
    "build_masks",
    "model_target_positions",
    "critical_target_positions",
    "prediction_positions",
    "validate_target_positions",
]
