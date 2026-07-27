"""Numerically stable primitives shared by the exact and bucketed objectives.

Everything here works in float32 regardless of the incoming dtype: BF16/FP16
logits are upcast before any ``log_softmax``/``logsumexp``, because a
152k-entry softmax reduction in half precision loses several digits of the tail
mass this project depends on.
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "LOG_PROB_FLOOR",
    "NEG_CLAMP",
    "neg_clamp_for",
    "to_float32",
    "log_softmax_f32",
    "log1mexp",
    "safe_log_prob",
    "kl_from_log_probs",
    "entropy_from_log_probs",
]

#: Finite stand-in for ``-inf`` log-probabilities.  ``exp(LOG_PROB_FLOOR)`` is
#: exactly ``0.0`` in float32, so clamping to it never changes a probability,
#: but it keeps every difference ``log p - log q`` finite and therefore keeps
#: gradients free of ``inf - inf = nan``.
LOG_PROB_FLOOR: float = -1.0e30

#: Fallback bound for ``log1mexp`` when the dtype is unknown.  ``log(1 - exp(x))``
#: diverges as ``x -> 0``, so the input is clamped; ``-1e-7`` is float32's
#: resolution near 1.0 and bounds the result near ``-16.1``.
NEG_CLAMP: float = -1.0e-7


def neg_clamp_for(dtype: torch.dtype) -> float:
    """Largest input ``log1mexp`` accepts for ``dtype``.

    The clamp is not arbitrary: it is the point past which ``1 - exp(x)`` stops
    being representable.  Using ``finfo(dtype).eps`` means float32 keeps the
    ~1e-7 bound it has always had while float64 resolves a tail mass a billion
    times smaller, so a test written in float64 measures the mathematics rather
    than a float32 constant.
    """
    if dtype in (torch.float32, torch.float64, torch.float16, torch.bfloat16):
        return -float(torch.finfo(dtype).eps)
    return NEG_CLAMP


_LOG_HALF = -math.log(2.0)


def to_float32(x: torch.Tensor) -> torch.Tensor:
    """Upcast to float32 unless already float32 or float64."""
    if x.dtype in (torch.float32, torch.float64):
        return x
    return x.to(torch.float32)


def log_softmax_f32(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """``log_softmax(logits / temperature)`` accumulated in float32."""
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    x = to_float32(logits)
    if temperature != 1.0:
        x = x / temperature
    return torch.log_softmax(x, dim=-1)


def log1mexp(x: torch.Tensor) -> torch.Tensor:
    """Compute ``log(1 - exp(x))`` for ``x <= 0``, stably and differentiably.

    Two regimes are used (Machler, 2012): ``log(-expm1(x))`` when ``x`` is close
    to zero and ``log1p(-exp(x))`` when it is far from zero.  Both branches are
    evaluated on *sanitized* inputs so the unused branch cannot inject ``nan``
    into the backward pass -- the usual failure mode of a naive
    ``torch.where`` over the raw tensor.

    The input is clamped at :func:`neg_clamp_for` for its dtype.  In float32
    that caps the recoverable tail mass at about ``1e-7``, so a teacher whose
    top-k captures all but ``1e-9`` of the mass is reported as capturing all but
    ``1e-7``.  That is a float32 resolution limit, not a choice, and it is why
    the effective reverse-KL tail penalty is bounded near 16.1 nats rather than
    the ``log(1 / tail_epsilon)`` that ``tail_epsilon`` alone would suggest.
    """
    x = x.clamp(max=neg_clamp_for(x.dtype))
    near_zero = x > _LOG_HALF
    safe_far = torch.full_like(x, _LOG_HALF - 1.0)
    x_near = torch.where(near_zero, x, safe_far)
    x_far = torch.where(near_zero, safe_far, x)
    out_near = torch.log(-torch.expm1(x_near))
    out_far = torch.log1p(-torch.exp(x_far))
    return torch.where(near_zero, out_near, out_far)


def safe_log_prob(log_p: torch.Tensor, floor: float = LOG_PROB_FLOOR) -> torch.Tensor:
    """Clamp a log-probability tensor away from ``-inf``."""
    return log_p.clamp_min(floor)


def kl_from_log_probs(log_p: torch.Tensor, log_q: torch.Tensor) -> torch.Tensor:
    """``KL(P || Q) = sum_v p_v (log p_v - log q_v)`` reduced over the last axis.

    ``log_p`` and ``log_q`` must be genuine log-probabilities (already
    normalized).  The name order is the *orientation*: the first argument is the
    distribution the expectation is taken under.
    """
    log_p = safe_log_prob(to_float32(log_p))
    log_q = safe_log_prob(to_float32(log_q))
    p = log_p.exp()
    return (p * (log_p - log_q)).sum(dim=-1)


def entropy_from_log_probs(log_p: torch.Tensor) -> torch.Tensor:
    """``H(P) = -sum_v p_v log p_v`` reduced over the last axis (nats)."""
    log_p = safe_log_prob(to_float32(log_p))
    p = log_p.exp()
    return -(p * log_p).sum(dim=-1)
