"""Versioned task-reward advantage composition."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

__all__ = [
    "ADVANTAGE_COMPOSER_VERSION",
    "AdvantageComposition",
    "AdvantageComposer",
    "AdvantageMode",
]

ADVANTAGE_COMPOSER_VERSION = "miniverl-task-distill-advantage-v1"


class AdvantageMode(str, Enum):
    NONE = "none"
    RAW = "raw"
    GROUP_CENTER = "group_center"
    GROUP_STANDARDIZE = "group_standardize"
    LEAVE_ONE_OUT = "leave_one_out"


@dataclass(frozen=True)
class AdvantageComposition:
    task_advantages: list[float]
    mode: AdvantageMode
    distillation_coef: float
    task_reward_coef: float
    zero_variance: bool
    implementation_version: str = ADVANTAGE_COMPOSER_VERSION


@dataclass(frozen=True)
class AdvantageComposer:
    mode: AdvantageMode
    distillation_coef: float = 1.0
    task_reward_coef: float = 1.0

    def __post_init__(self) -> None:
        for name, value in (
            ("distillation_coef", self.distillation_coef),
            ("task_reward_coef", self.task_reward_coef),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")

    def compose_group(self, rewards: list[float]) -> AdvantageComposition:
        if not rewards:
            raise ValueError("reward group cannot be empty")
        if any(not math.isfinite(value) for value in rewards):
            raise ValueError("reward group must contain only finite values")
        if self.mode is AdvantageMode.LEAVE_ONE_OUT and len(rewards) < 2:
            raise ValueError("leave_one_out requires at least two samples per prompt")
        mean = sum(rewards) / len(rewards)
        variance = sum((value - mean) ** 2 for value in rewards) / len(rewards)
        zero_variance = variance == 0.0
        if self.mode is AdvantageMode.NONE:
            advantages = [0.0] * len(rewards)
        elif self.mode is AdvantageMode.RAW:
            advantages = list(rewards)
        elif self.mode is AdvantageMode.GROUP_CENTER:
            advantages = [value - mean for value in rewards]
        elif self.mode is AdvantageMode.GROUP_STANDARDIZE:
            scale = math.sqrt(variance)
            advantages = (
                [0.0] * len(rewards)
                if zero_variance
                else [(value - mean) / scale for value in rewards]
            )
        else:
            advantages = [value - (sum(rewards) - value) / (len(rewards) - 1) for value in rewards]
        return AdvantageComposition(
            task_advantages=advantages,
            mode=self.mode,
            distillation_coef=self.distillation_coef,
            task_reward_coef=self.task_reward_coef,
            zero_variance=zero_variance,
        )

    def expand_task_advantage(self, value: float, assistant_mask: list[bool]) -> list[float]:
        if not math.isfinite(value):
            raise ValueError("task advantage must be finite")
        return [value if selected else 0.0 for selected in assistant_mask]
