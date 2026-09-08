"""Clipped policy objective matching verl v0.9.0 vanilla policy loss."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PolicyLossOutput:
    loss: torch.Tensor
    per_token_loss: torch.Tensor
    ratio: torch.Tensor
    metrics: dict[str, float]


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.where(mask.bool(), values, 0.0).sum() / (mask.sum() + 1e-8)


def clipped_policy_loss(
    *,
    current_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    clip_ratio: float = 0.2,
    clip_ratio_low: float | None = None,
    clip_ratio_high: float | None = None,
    clip_ratio_c: float = 3.0,
) -> PolicyLossOutput:
    """Return token-mean dual-clipped PPO/GRPO policy loss."""
    shape = current_log_probs.shape
    if any(value.shape != shape for value in (old_log_probs, advantages, response_mask)):
        raise ValueError("policy tensors must have identical shape")
    if not all(
        torch.isfinite(value).all() for value in (current_log_probs, old_log_probs, advantages)
    ):
        raise ValueError("policy log-probabilities and advantages must be finite")
    if not bool(response_mask.sum() > 0):
        raise ValueError("response_mask must select at least one token")
    if clip_ratio_c <= 1:
        raise ValueError("clip_ratio_c must be greater than one")
    low = clip_ratio if clip_ratio_low is None else clip_ratio_low
    high = clip_ratio if clip_ratio_high is None else clip_ratio_high
    negative_approx_kl = (current_log_probs - old_log_probs).clamp(-20.0, 20.0)
    ratio = torch.exp(negative_approx_kl)
    losses1 = -advantages * ratio
    losses2 = -advantages * ratio.clamp(1 - low, 1 + high)
    clipped = torch.maximum(losses1, losses2)
    dual = torch.minimum(-advantages * clip_ratio_c, clipped)
    per_token = torch.where(advantages < 0, dual, clipped)
    loss = _masked_mean(per_token, response_mask)
    metrics = {
        "pg_clipfrac": float(_masked_mean((losses2 > losses1).float(), response_mask).detach()),
        "ppo_kl": float(_masked_mean(-negative_approx_kl, response_mask).detach()),
        "pg_clipfrac_lower": float(
            _masked_mean(
                ((clipped > -advantages * clip_ratio_c) & (advantages < 0)).float(), response_mask
            ).detach()
        ),
        "ratio_mean": float(_masked_mean(ratio, response_mask).detach()),
    }
    return PolicyLossOutput(loss=loss, per_token_loss=per_token, ratio=ratio, metrics=metrics)
