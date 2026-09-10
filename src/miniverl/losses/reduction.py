"""Masked reductions and explicit logical-batch RL aggregation contracts."""

from __future__ import annotations

import torch


def rl_reduction_weights(
    rows: list[torch.Tensor],
    aggregation: str,
    *,
    response_length: int,
    loss_scale_factor: float | None = None,
) -> tuple[list[torch.Tensor], float]:
    """Lower pinned verl reductions to weights plus a logical-batch denominator.

    Compute once before physical partitioning. The response horizon is the
    upstream padded response width, not the local microbatch's longest sample.
    Fully masked sequences do not contribute to the sequence count.
    """
    counts = [float(row.sum()) for row in rows]
    if not sum(counts) > 0:
        raise ValueError("RL reduction must select at least one token")
    if any(not torch.isfinite(row).all() or bool((row < 0).any()) for row in rows):
        raise ValueError("RL reduction weights must be finite and non-negative")
    sequences = float(sum(count > 0 for count in counts))
    if aggregation == "token-mean":
        return rows, sum(counts)
    if aggregation == "token-sum":
        return rows, 1.0
    if aggregation == "seq-mean-token-sum":
        return rows, sequences
    if aggregation == "seq-mean-token-mean":
        return [row / (count + 1e-8) for row, count in zip(rows, counts, strict=True)], sequences
    if aggregation == "seq-mean-token-sum-norm":
        scale = float(response_length if loss_scale_factor is None else loss_scale_factor)
        if not torch.isfinite(torch.tensor(scale)) or scale <= 0:
            raise ValueError("loss scale factor must be finite and positive")
        return rows, sequences * scale
    raise ValueError(f"unsupported RL aggregation: {aggregation}")


__all__ = ["MIN_TOTAL_WEIGHT", "total_weight", "weighted_mean"]

#: Denominator substituted when the weights sum to exactly zero.  The numerator
#: is then exactly zero too, so the loss is 0 and the gradient is 0.
MIN_TOTAL_WEIGHT: float = 1e-12

#: Accumulation dtype for the two reductions below.  These operate on ``[N]``
#: per-position tensors -- a few hundred elements, never a vocabulary-sized one --
#: so the wider accumulator costs nothing measurable even on a consumer card
#: where float64 runs at a fraction of float32 throughput.  It buys exactness
#: across the whole float32 input range: with a weight of ``1.4e-45`` (float32's
#: smallest subnormal) the product ``1.5 * 1.4e-45`` rounds to *twice* the
#: subnormal minimum in float32, so ``sum(w*x)/sum(w)`` returns 2.0 for a single
#: value of 1.5 -- a result outside the range of its own inputs.  float64 has
#: ~260 decades of headroom below that point and reproduces 1.5 exactly.
_ACCUM = torch.float64


def total_weight(weights: torch.Tensor) -> torch.Tensor:
    """Sum of token weights as a float32 scalar tensor.

    Accumulated in float64 for the reason given at :data:`_ACCUM`, then returned
    as float32 so the public dtype contract is unchanged.
    """
    return weights.to(_ACCUM).sum().to(torch.float32)


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
    w = weights.to(_ACCUM)
    x = per_token.to(_ACCUM)
    if denominator is None:
        denom = w.sum()
    elif isinstance(denominator, torch.Tensor):
        denom = denominator.to(_ACCUM)
    else:
        denom = torch.as_tensor(float(denominator), dtype=_ACCUM, device=x.device)
    # Substitute the floor only when the weights sum to *exactly* zero. Clamping
    # instead would silently rescale the result for any tiny-but-positive weight
    # sum: with weights summing to 2e-16 the true mean is still well defined and
    # bounded by max(x), so dividing by the real total is both correct and safe.
    denom = torch.where(denom > 0, denom, torch.full_like(denom, MIN_TOTAL_WEIGHT))
    return ((x * w).sum() / denom).to(torch.float32)
