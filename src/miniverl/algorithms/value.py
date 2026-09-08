"""Clipped critic objective matching verl's PPO value loss."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ValueLossOutput:
    loss: torch.Tensor
    per_token_loss: torch.Tensor
    metrics: dict[str, float]


def clipped_value_loss(
    current_values: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    cliprange_value: float = 0.5,
) -> ValueLossOutput:
    """Compute the token-mean maximum of unclipped and clipped value errors."""
    shape = current_values.shape
    if any(value.shape != shape for value in (old_values, returns, response_mask)):
        raise ValueError("value tensors must have identical shape")
    if not all(torch.isfinite(value).all() for value in (current_values, old_values, returns)):
        raise ValueError("values and returns must be finite")
    count = response_mask.sum()
    if not bool(count > 0):
        raise ValueError("response_mask must select at least one token")
    clipped = old_values + (current_values - old_values).clamp(-cliprange_value, cliprange_value)
    loss_unclipped = (current_values - returns).square()
    loss_clipped = (clipped - returns).square()
    per_token = 0.5 * torch.maximum(loss_unclipped, loss_clipped)
    loss = torch.where(response_mask.bool(), per_token, 0.0).sum() / (count + 1e-8)
    clipfrac = torch.where(
        response_mask.bool(), (loss_clipped > loss_unclipped).float(), 0.0
    ).sum() / (count + 1e-8)
    return ValueLossOutput(
        loss=loss,
        per_token_loss=per_token,
        metrics={
            "value_loss": float(loss.detach()),
            "value_clipfrac": float(clipfrac.detach()),
            "value_mean": float(
                (
                    torch.where(response_mask.bool(), current_values, 0.0).sum() / (count + 1e-8)
                ).detach()
            ),
        },
    )
