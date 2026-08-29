"""Pinned verl v0.8 sampled-k1 policy-gradient objective.

This module implements only the closed
``verl-opd-v0.8-single-gpu-pg-k1-v1`` contract.  It is deliberately not a
general PPO implementation: task rewards, external advantages, rollout
importance weights, critics, and alternative policy losses are outside the
profile.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from miniverl.bridge.opd_pg_contract import VERL_PG_K1_IMPLEMENTATION_VERSION

__all__ = [
    "VERL_PG_K1_IMPLEMENTATION_VERSION",
    "VerlPGK1Output",
    "RewardedPGK1Output",
    "verl_k1_estimator",
    "verl_pg_k1_loss",
    "rewarded_pg_k1_loss",
]


@dataclass(frozen=True)
class VerlPGK1Output:
    """Exact tensors and scalar produced by the supported pinned path."""

    loss: torch.Tensor
    estimator: torch.Tensor
    advantages: torch.Tensor
    per_token_loss: torch.Tensor
    ratio: torch.Tensor
    metrics: dict[str, float]


@dataclass(frozen=True)
class RewardedPGK1Output:
    """Explicit distillation and task components for the rewarded profile."""

    loss: torch.Tensor
    estimator: torch.Tensor
    distillation_advantages: torch.Tensor
    task_advantages: torch.Tensor
    advantages: torch.Tensor
    per_token_loss: torch.Tensor
    ratio: torch.Tensor
    metrics: dict[str, float]


def _require_vector(name: str, value: torch.Tensor, length: int | None = None) -> None:
    if value.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional token vector")
    if length is not None and value.numel() != length:
        raise ValueError(f"{name} has {value.numel()} tokens, expected {length}")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")


def verl_k1_estimator(
    old_actor_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Return the pinned sampled k1 estimate: ``old_actor - teacher``."""
    _require_vector("old_actor_log_probs", old_actor_log_probs)
    _require_vector("teacher_log_probs", teacher_log_probs, old_actor_log_probs.numel())
    return old_actor_log_probs.to(torch.float32) - teacher_log_probs.to(torch.float32)


def _masked_mean(value: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    total = weights.sum()
    if not bool(total > 0):
        raise ValueError("weights must select at least one response token")
    return (value * weights).sum() / total


def verl_pg_k1_loss(
    *,
    current_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    weights: torch.Tensor,
    clip_ratio: float = 0.2,
    clip_ratio_low: float | None = None,
    clip_ratio_high: float | None = None,
    clip_ratio_c: float = 3.0,
    loss_max_clamp: float | None = None,
) -> VerlPGK1Output:
    """Apply verl v0.8 k1 -> detached advantage -> vanilla policy loss.

    ``weights`` is miniVERL's flattened response mask.  The supported profile
    requires unit weights, but accepting a numeric mask here keeps the loss
    primitive independently testable and exactly reproduces token-mean
    aggregation.
    """
    n = current_log_probs.numel()
    _require_vector("current_log_probs", current_log_probs)
    _require_vector("old_log_probs", old_log_probs, n)
    _require_vector("teacher_log_probs", teacher_log_probs, n)
    _require_vector("weights", weights, n)
    if bool((weights < 0).any()):
        raise ValueError("weights must be non-negative")
    if not 0.0 <= clip_ratio < 1.0:
        raise ValueError("clip_ratio must be in [0, 1)")
    low = clip_ratio if clip_ratio_low is None else clip_ratio_low
    high = clip_ratio if clip_ratio_high is None else clip_ratio_high
    if not 0.0 <= low < 1.0 or not 0.0 <= high < 1.0:
        raise ValueError("clip ratios must be in [0, 1)")
    if clip_ratio_c <= 1.0:
        raise ValueError("clip_ratio_c must be greater than 1")
    if loss_max_clamp is not None and loss_max_clamp <= 0.0:
        raise ValueError("loss_max_clamp must be positive")

    work_weights = weights.to(device=current_log_probs.device, dtype=torch.float32)
    estimator = verl_k1_estimator(old_log_probs, teacher_log_probs).to(current_log_probs.device)
    if loss_max_clamp is not None:
        estimator = estimator.clamp(min=-loss_max_clamp, max=loss_max_clamp)
    advantages = -estimator.detach()

    negative_approx_kl = (
        current_log_probs.to(torch.float32) - old_log_probs.to(torch.float32)
    ).clamp(min=-20.0, max=20.0)
    ratio = negative_approx_kl.exp()
    losses1 = -advantages * ratio
    losses2 = -advantages * ratio.clamp(1.0 - low, 1.0 + high)
    clipped = torch.maximum(losses1, losses2)
    dual_clipped = torch.minimum(-advantages * clip_ratio_c, clipped)
    per_token_loss = torch.where(advantages < 0, dual_clipped, clipped)
    loss = _masked_mean(per_token_loss, work_weights)

    metrics = {
        "pg_clipfrac": float(
            _masked_mean((losses2 > losses1).to(torch.float32), work_weights).detach()
        ),
        "ppo_kl": float(_masked_mean(-negative_approx_kl, work_weights).detach()),
        "pg_clipfrac_lower": float(
            _masked_mean(
                ((clipped > -advantages * clip_ratio_c) & (advantages < 0)).to(torch.float32),
                work_weights,
            ).detach()
        ),
        "abs_k1": float(_masked_mean(estimator.abs(), work_weights).detach()),
    }
    return VerlPGK1Output(
        loss=loss,
        estimator=estimator,
        advantages=advantages,
        per_token_loss=per_token_loss,
        ratio=ratio,
        metrics=metrics,
    )


def rewarded_pg_k1_loss(
    *,
    current_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    weights: torch.Tensor,
    task_advantage: float,
    distillation_coef: float,
    task_reward_coef: float,
    clip_ratio: float = 0.2,
    clip_ratio_low: float | None = None,
    clip_ratio_high: float | None = None,
    clip_ratio_c: float = 3.0,
    loss_max_clamp: float | None = None,
) -> RewardedPGK1Output:
    """Compose task and distillation advantages for the rewarded PG profile."""

    n = current_log_probs.numel()
    _require_vector("current_log_probs", current_log_probs)
    _require_vector("old_log_probs", old_log_probs, n)
    _require_vector("teacher_log_probs", teacher_log_probs, n)
    _require_vector("weights", weights, n)
    if bool((weights < 0).any()):
        raise ValueError("weights must be non-negative")
    if not torch.isfinite(torch.tensor(task_advantage)):
        raise ValueError("task_advantage must be finite")
    if not torch.isfinite(torch.tensor([distillation_coef, task_reward_coef])).all() or (
        distillation_coef < 0 or task_reward_coef < 0
    ):
        raise ValueError("advantage coefficients must be finite and non-negative")
    if not 0.0 <= clip_ratio < 1.0:
        raise ValueError("clip_ratio must be in [0, 1)")
    low = clip_ratio if clip_ratio_low is None else clip_ratio_low
    high = clip_ratio if clip_ratio_high is None else clip_ratio_high
    if not 0.0 <= low < 1.0 or not 0.0 <= high < 1.0:
        raise ValueError("clip ratios must be in [0, 1)")
    if clip_ratio_c <= 1.0:
        raise ValueError("clip_ratio_c must be greater than 1")

    work_weights = weights.to(device=current_log_probs.device, dtype=torch.float32)
    estimator = verl_k1_estimator(old_log_probs, teacher_log_probs).to(current_log_probs.device)
    if loss_max_clamp is not None:
        estimator = estimator.clamp(min=-loss_max_clamp, max=loss_max_clamp)
    distillation_advantages = -estimator.detach()
    task_advantages = torch.full_like(distillation_advantages, float(task_advantage))
    advantages = (
        distillation_coef * distillation_advantages + task_reward_coef * task_advantages
    ).detach()
    negative_approx_kl = (
        current_log_probs.to(torch.float32) - old_log_probs.to(torch.float32)
    ).clamp(min=-20.0, max=20.0)
    ratio = negative_approx_kl.exp()
    losses1 = -advantages * ratio
    losses2 = -advantages * ratio.clamp(1.0 - low, 1.0 + high)
    clipped = torch.maximum(losses1, losses2)
    dual_clipped = torch.minimum(-advantages * clip_ratio_c, clipped)
    per_token_loss = torch.where(advantages < 0, dual_clipped, clipped)
    loss = _masked_mean(per_token_loss, work_weights)
    metrics = {
        "distillation_advantage_mean": float(
            _masked_mean(distillation_advantages, work_weights).detach()
        ),
        "task_advantage_mean": float(_masked_mean(task_advantages, work_weights).detach()),
        "total_advantage_mean": float(_masked_mean(advantages, work_weights).detach()),
        "pg_clipfrac": float(
            _masked_mean((losses2 > losses1).to(torch.float32), work_weights).detach()
        ),
        "ppo_kl": float(_masked_mean(-negative_approx_kl, work_weights).detach()),
    }
    return RewardedPGK1Output(
        loss=loss,
        estimator=estimator,
        distillation_advantages=distillation_advantages,
        task_advantages=task_advantages,
        advantages=advantages,
        per_token_loss=per_token_loss,
        ratio=ratio,
        metrics=metrics,
    )
