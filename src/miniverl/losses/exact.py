"""Exact full-vocabulary divergences.

These functions materialize the complete ``[N, V]`` teacher and student
distributions for ``N`` selected prediction positions.  That is affordable when
``N`` is a chunk (a few hundred) rather than a full batch of sequences, which
is exactly how :mod:`miniverl.losses.chunked` calls them.

Orientation is explicit in every name.  ``teacher`` is the reference
distribution ``P``; ``student`` is the trainable distribution ``Q``.

* ``forward_kl``  = ``KL(teacher || student)``  -- mass-covering
* ``reverse_kl``  = ``KL(student || teacher)``  -- mode-seeking, the usual
  on-policy distillation objective
* ``jsd``         = the beta-weighted Jensen-Shannon divergence

Temperature
-----------
Both distributions are softmaxed at ``temperature``.  When
``scale_by_temperature_squared`` is set the divergence is multiplied by
``T**2``.  The classic Hinton et al. (2015) high-temperature derivation applies
to forward KL / soft-target cross-entropy near the uniform regime.  miniVERL
offers the same factor for reverse KL and JSD as an explicit heuristic; it does
not claim temperature-invariant gradients for those objectives or for sharply
peaked distributions.  ``scripts/temperature_gradient_sweep.py`` measures all
three cases instead of generalizing the forward-KL derivation.
"""

from __future__ import annotations

import math

import torch

from miniverl.losses.numerics import (
    entropy_from_log_probs,
    kl_from_log_probs,
    log_softmax_f32,
    safe_log_prob,
    to_float32,
)

__all__ = [
    "exact_forward_kl",
    "exact_reverse_kl",
    "exact_jsd",
    "exact_divergence",
    "exact_teacher_entropy",
    "temperature_scale",
]


def temperature_scale(
    value: torch.Tensor, temperature: float, scale_by_temperature_squared: bool
) -> torch.Tensor:
    """Apply the optional ``T**2`` factor (heuristic outside high-T forward KL)."""
    if scale_by_temperature_squared and temperature != 1.0:
        return value * (temperature**2)
    return value


def exact_forward_kl(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    scale_by_temperature_squared: bool = True,
) -> torch.Tensor:
    """``KL(teacher || student)`` per position, shape ``[N]``."""
    teacher_log_probs = log_softmax_f32(teacher_logits, temperature)
    student_log_probs = log_softmax_f32(student_logits, temperature)
    value = kl_from_log_probs(teacher_log_probs, student_log_probs)
    return temperature_scale(value, temperature, scale_by_temperature_squared)


def exact_reverse_kl(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    scale_by_temperature_squared: bool = True,
) -> torch.Tensor:
    """``KL(student || teacher)`` per position, shape ``[N]``."""
    teacher_log_probs = log_softmax_f32(teacher_logits, temperature)
    student_log_probs = log_softmax_f32(student_logits, temperature)
    value = kl_from_log_probs(student_log_probs, teacher_log_probs)
    return temperature_scale(value, temperature, scale_by_temperature_squared)


def exact_jsd(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    *,
    beta: float = 0.5,
    temperature: float = 1.0,
    scale_by_temperature_squared: bool = True,
) -> torch.Tensor:
    """Beta-weighted Jensen-Shannon divergence per position, shape ``[N]``.

    ``M = beta * teacher + (1 - beta) * student`` and

    ``JS_beta = beta * KL(teacher || M) + (1 - beta) * KL(student || M)``.

    ``beta = 0.5`` is the symmetric Jensen-Shannon divergence.  ``beta`` must be
    strictly inside ``(0, 1)``: at either endpoint ``M`` collapses onto one of
    the two distributions and ``JS_beta`` is identically zero (the KL limits are
    recovered only as ``JS_beta / beta``, not as ``JS_beta`` itself).
    """
    if not 0.0 < beta < 1.0:
        raise ValueError(
            f"jsd beta must be strictly inside (0, 1); got {beta}. "
            "At beta=0 or beta=1 the mixture equals one of the inputs and the "
            "divergence is identically zero."
        )
    teacher_log_probs = safe_log_prob(log_softmax_f32(teacher_logits, temperature))
    student_log_probs = safe_log_prob(log_softmax_f32(student_logits, temperature))
    log_beta = math.log(beta)
    log_one_minus_beta = math.log1p(-beta)
    log_mixture = torch.logaddexp(
        teacher_log_probs + log_beta, student_log_probs + log_one_minus_beta
    )
    value = beta * kl_from_log_probs(teacher_log_probs, log_mixture) + (
        1.0 - beta
    ) * kl_from_log_probs(student_log_probs, log_mixture)
    return temperature_scale(value, temperature, scale_by_temperature_squared)


def exact_divergence(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    *,
    divergence: str,
    temperature: float = 1.0,
    scale_by_temperature_squared: bool = True,
    jsd_beta: float = 0.5,
) -> torch.Tensor:
    """Dispatch to the exact divergence named by ``divergence``."""
    if divergence == "forward_kl":
        return exact_forward_kl(
            teacher_logits,
            student_logits,
            temperature=temperature,
            scale_by_temperature_squared=scale_by_temperature_squared,
        )
    if divergence == "reverse_kl":
        return exact_reverse_kl(
            teacher_logits,
            student_logits,
            temperature=temperature,
            scale_by_temperature_squared=scale_by_temperature_squared,
        )
    if divergence == "jsd":
        return exact_jsd(
            teacher_logits,
            student_logits,
            beta=jsd_beta,
            temperature=temperature,
            scale_by_temperature_squared=scale_by_temperature_squared,
        )
    raise ValueError(f"unknown divergence {divergence!r}; expected forward_kl, reverse_kl or jsd")


def exact_teacher_entropy(
    teacher_logits: torch.Tensor, *, temperature: float = 1.0
) -> torch.Tensor:
    """Teacher entropy in nats per position, shape ``[N]``."""
    return entropy_from_log_probs(log_softmax_f32(to_float32(teacher_logits), temperature))
