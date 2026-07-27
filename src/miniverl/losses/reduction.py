"""Masked, weight-normalized reduction.

Every miniVERL objective is normalized by **the sum of effective token
weights**, never by the padded sequence length and never by the raw position
count.  Two runs with different selection budgets therefore produce comparable
loss magnitudes, and a masked token contributes exactly zero -- not "almost
zero because it was averaged over a larger denominator".
"""

from __future__ import annotations

import torch

__all__ = ["MIN_TOTAL_WEIGHT", "total_weight", "weighted_mean"]

#: Denominator floor.  Only reachable when every selected token has zero
#: weight, in which case the numerator is exactly zero too and the loss is 0.
MIN_TOTAL_WEIGHT: float = 1e-12


def total_weight(weights: torch.Tensor) -> torch.Tensor:
    """Sum of token weights as a float32 scalar tensor."""
    return weights.to(torch.float32).sum()


def weighted_mean(
    per_token: torch.Tensor,
    weights: torch.Tensor,
    *,
    denominator: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """``sum(w * x) / sum(w)`` with a safe denominator.

    Parameters
    ----------
    per_token:
        ``[N]`` per-position values.
    weights:
        ``[N]`` non-negative weights; zeros mask a position out exactly.
    denominator:
        Optional externally computed normalizer.  Chunked callers pass the
        **global** weight sum so that summing per-chunk contributions
        reproduces the unchunked result bit-for-bit in exact arithmetic.
    """
    if per_token.shape != weights.shape:
        raise ValueError(
            f"per_token shape {tuple(per_token.shape)} != weights shape {tuple(weights.shape)}"
        )
    w = weights.to(torch.float32)
    x = per_token.to(torch.float32)
    if denominator is None:
        denom = w.sum()
    elif isinstance(denominator, torch.Tensor):
        denom = denominator.to(torch.float32)
    else:
        denom = torch.as_tensor(float(denominator), dtype=torch.float32, device=x.device)
    denom = torch.clamp(denom, min=MIN_TOTAL_WEIGHT)
    return (x * w).sum() / denom
