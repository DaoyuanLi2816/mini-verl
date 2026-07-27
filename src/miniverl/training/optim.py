"""Optimizer and learning-rate schedule.

The schedule is a plain function of the step index rather than a
``torch.optim.lr_scheduler`` object.  That keeps its state a two-integer JSON
blob, which is what lets checkpoints stay pickle-free and resume exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from miniverl.config.models import LRSchedule, OptimizerName, TrainConfig
from miniverl.errors import MissingDependencyError
from miniverl.utils.lazy import have_module

__all__ = ["LearningRateSchedule", "build_optimizer"]


@dataclass
class LearningRateSchedule:
    """Constant / linear / cosine decay with linear warmup."""

    kind: LRSchedule
    base_lr: float
    warmup_steps: int
    total_steps: int
    min_lr_ratio: float = 0.0

    def lr_at(self, step: int) -> float:
        """Learning rate for ``step`` (0-based)."""
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return self.base_lr * (step + 1) / self.warmup_steps
        if self.kind is LRSchedule.CONSTANT:
            return self.base_lr
        decay_steps = max(self.total_steps - self.warmup_steps, 1)
        progress = min(max((step - self.warmup_steps) / decay_steps, 0.0), 1.0)
        if self.kind is LRSchedule.LINEAR:
            factor = 1.0 - progress
        else:
            factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.base_lr * (self.min_lr_ratio + (1.0 - self.min_lr_ratio) * factor)

    def state_dict(self) -> dict[str, Any]:
        """Serializable schedule state."""
        return {
            "kind": self.kind.value,
            "base_lr": self.base_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "min_lr_ratio": self.min_lr_ratio,
        }

    @classmethod
    def from_state_dict(cls, payload: dict[str, Any]) -> LearningRateSchedule:
        """Rebuild from :meth:`state_dict`."""
        return cls(
            kind=LRSchedule(payload["kind"]),
            base_lr=float(payload["base_lr"]),
            warmup_steps=int(payload["warmup_steps"]),
            total_steps=int(payload["total_steps"]),
            min_lr_ratio=float(payload.get("min_lr_ratio", 0.0)),
        )


def build_optimizer(parameters: list[Any], train: TrainConfig) -> Any:
    """Construct the configured optimizer over ``parameters``."""
    import torch

    if not parameters:
        raise ValueError(
            "the student has no trainable parameters; check models.student.lora.enabled "
            "and models.student.lora.target_modules"
        )
    kwargs: dict[str, Any] = {
        "lr": train.learning_rate,
        "betas": (train.adam_beta1, train.adam_beta2),
        "eps": train.adam_eps,
        "weight_decay": train.weight_decay,
    }
    if train.optimizer is OptimizerName.ADAMW_8BIT:
        if not have_module("bitsandbytes"):
            raise MissingDependencyError("bitsandbytes", "cuda", "the adamw8bit optimizer")
        import bitsandbytes as bnb

        return bnb.optim.AdamW8bit(parameters, **kwargs)
    return torch.optim.AdamW(parameters, **kwargs)
