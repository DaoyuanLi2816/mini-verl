"""Training loops, memory strategies and checkpointing."""

from __future__ import annotations

from miniverl.training.checkpoint import (
    CheckpointState,
    ValidatedCheckpoint,
    latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
    validate_checkpoint,
)
from miniverl.training.memory import MemoryPlan, resolve_strategy, run_with_oom_retry
from miniverl.training.optim import LearningRateSchedule, build_optimizer
from miniverl.training.trainer import OPDTrainer, TrainResult, TrainSample

__all__ = [
    "OPDTrainer",
    "TrainResult",
    "TrainSample",
    "MemoryPlan",
    "resolve_strategy",
    "run_with_oom_retry",
    "CheckpointState",
    "ValidatedCheckpoint",
    "save_checkpoint",
    "load_checkpoint",
    "validate_checkpoint",
    "latest_checkpoint",
    "LearningRateSchedule",
    "build_optimizer",
]
