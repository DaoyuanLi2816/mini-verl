"""Reference-policy KL estimators matching pinned verl v0.9.0."""

from __future__ import annotations

from typing import Literal

import torch

KLPenalty = Literal["kl", "abs", "mse", "low_var_kl"]


def reference_kl_penalty(
    log_probs: torch.Tensor,
    reference_log_probs: torch.Tensor,
    *,
    penalty: KLPenalty = "kl",
) -> torch.Tensor:
    """Return verl's sampled-token KL estimator for a fixed reference policy."""
    if log_probs.shape != reference_log_probs.shape:
        raise ValueError("policy and reference log-probabilities must have identical shape")
    if not torch.isfinite(log_probs).all() or not torch.isfinite(reference_log_probs).all():
        raise ValueError("policy and reference log-probabilities must be finite")
    difference = log_probs - reference_log_probs
    if penalty == "kl":
        return difference
    if penalty == "abs":
        return difference.abs()
    if penalty == "mse":
        return 0.5 * difference.square()
    if penalty == "low_var_kl":
        reverse = (-difference).clamp(-20.0, 20.0)
        return (torch.exp(reverse) - reverse - 1.0).clamp(-10.0, 10.0)
    raise ValueError(f"unsupported reference KL penalty {penalty!r}")
