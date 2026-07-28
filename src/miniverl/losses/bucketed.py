"""Compressed ``top-k + tail`` divergences.

What this is
------------
The teacher's distribution over ``V`` vocabulary entries is *coarse-grained*
into ``K + 1`` buckets: one bucket per teacher top-k token, plus a single
aggregate bucket holding all remaining mass.  The student distribution is
coarse-grained the same way -- its probabilities on the teacher's top-k tokens,
plus ``1 - sum_k q`` in the tail.  The divergence is then computed between the
two ``K + 1`` category distributions.

What this is **not**
--------------------
This is *not* full-vocabulary KL, and the functions are named
``bucketed_*`` so no call site can pretend otherwise.  By the data-processing
inequality the *unfloored mathematical coarse-graining* is a lower bound on the
exact full-vocabulary divergence: coarse-graining can only destroy information.
The implemented epsilon-smoothed objective is finite and numerically robust,
but the floor means the theorem is not claimed literally for every input.
When ``K == V`` the empty tail bypasses smoothing and equality is exact up to
the arithmetic of the shared full-vocabulary primitives.

What it actually saves
----------------------
Teacher-side storage and transfer: ``K`` indices + ``K`` log-probs + one tail
scalar per position, instead of ``V`` logits.  It also lets the teacher be
evicted from VRAM between the scoring phase and the update phase.  It does
**not** reduce the student forward/backward cost, which still needs a
full-vocabulary ``log_softmax`` over the selected positions to normalize
correctly.

Tail handling
-------------
Both tails are floored at ``log(tail_epsilon)`` before use.  Without the floor,
a teacher whose top-k captures the entire mass (``p_tail = 0``) combined with a
student that still leaks probability outside the top-k produces ``+inf`` in
reverse KL.  The floor turns that into a bounded penalty of at most
``log(1 / tail_epsilon)`` nats.  After flooring, both category vectors are
renormalized so they are exact probability distributions and the divergence is
guaranteed non-negative.
"""

from __future__ import annotations

import math

import torch

from miniverl.losses.exact import temperature_scale
from miniverl.losses.numerics import (
    entropy_from_log_probs,
    kl_from_log_probs,
    log1mexp,
    log_softmax_f32,
    to_float32,
)

__all__ = [
    "teacher_topk_targets",
    "student_bucket_log_probs",
    "build_bucket_distributions",
    "bucketed_forward_kl",
    "bucketed_reverse_kl",
    "bucketed_jsd",
    "bucketed_divergence",
    "bucketed_teacher_entropy",
]


def teacher_topk_targets(
    teacher_logits: torch.Tensor,
    *,
    top_k: int,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compress teacher logits into ``(indices, topk_log_probs, tail_log_prob)``.

    ``topk_log_probs`` are log-probabilities over the **full** vocabulary
    restricted to the top-k ids -- they do not sum to one.  ``tail_log_prob``
    is ``log(1 - sum_k p)`` and is ``-inf``-safe via :func:`log1mexp`.

    Parameters
    ----------
    teacher_logits:
        ``[N, V]`` logits at the selected prediction positions.
    top_k:
        Number of vocabulary entries to keep.  Clipped to ``V``.
    temperature:
        Softmax temperature applied before compression; the cache records it so
        the consumer can reject a mismatched objective.
    """
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")
    log_probs = log_softmax_f32(teacher_logits, temperature)
    vocab_size = log_probs.shape[-1]
    k = min(top_k, vocab_size)
    topk_log_probs, topk_indices = torch.topk(log_probs, k=k, dim=-1)
    covered = torch.logsumexp(topk_log_probs, dim=-1)
    tail_log_prob = log1mexp(covered)
    if k == vocab_size:
        # The top-k is the whole vocabulary: the tail is exactly empty.  Say so
        # exactly instead of relying on floating-point cancellation.
        tail_log_prob = torch.full_like(tail_log_prob, float("-inf"))
    return topk_indices, topk_log_probs, tail_log_prob


def student_bucket_log_probs(
    student_logits: torch.Tensor,
    topk_indices: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Student log-probabilities on the teacher's buckets.

    Returns ``(topk_log_probs [N, K], tail_log_prob [N])``.  The student is
    normalized over the **full** vocabulary first -- that is what makes the tail
    bucket meaningful.
    """
    log_probs = log_softmax_f32(student_logits, temperature)
    topk = torch.gather(log_probs, dim=-1, index=topk_indices)
    covered = torch.logsumexp(topk, dim=-1)
    if topk_indices.shape[-1] == log_probs.shape[-1]:
        tail = torch.full_like(covered, float("-inf"))
    else:
        tail = log1mexp(covered)
    return topk, tail


def build_bucket_distributions(
    teacher_topk_log_probs: torch.Tensor,
    teacher_tail_log_prob: torch.Tensor,
    student_topk_log_probs: torch.Tensor,
    student_tail_log_prob: torch.Tensor,
    *,
    tail_epsilon: float = 1e-9,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Floor the tails, concatenate and renormalize into ``[N, K+1]`` log-probs."""
    if not 0.0 < tail_epsilon < 1.0:
        raise ValueError(f"tail_epsilon must be in (0, 1), got {tail_epsilon}")
    if bool(torch.isneginf(teacher_tail_log_prob).all()) and bool(
        torch.isneginf(student_tail_log_prob).all()
    ):
        # K == V: the partition is the identity.  Do not manufacture an
        # epsilon tail, because doing so scales the exact divergence by
        # 1/(1+epsilon) and breaks the advertised full-vocabulary limit.
        teacher = to_float32(teacher_topk_log_probs)
        student = to_float32(student_topk_log_probs)
        return (
            teacher - torch.logsumexp(teacher, dim=-1, keepdim=True),
            student - torch.logsumexp(student, dim=-1, keepdim=True),
        )
    log_eps = math.log(tail_epsilon)
    teacher = torch.cat(
        [
            to_float32(teacher_topk_log_probs),
            to_float32(teacher_tail_log_prob).clamp_min(log_eps).unsqueeze(-1),
        ],
        dim=-1,
    )
    student = torch.cat(
        [
            to_float32(student_topk_log_probs),
            to_float32(student_tail_log_prob).clamp_min(log_eps).unsqueeze(-1),
        ],
        dim=-1,
    )
    teacher = teacher - torch.logsumexp(teacher, dim=-1, keepdim=True)
    student = student - torch.logsumexp(student, dim=-1, keepdim=True)
    return teacher, student


def bucketed_forward_kl(
    teacher_bucket_log_probs: torch.Tensor,
    student_bucket_log_probs: torch.Tensor,
    *,
    temperature: float = 1.0,
    scale_by_temperature_squared: bool = True,
) -> torch.Tensor:
    """``KL(teacher || student)`` over the ``top-k + tail`` partition."""
    value = kl_from_log_probs(teacher_bucket_log_probs, student_bucket_log_probs)
    return temperature_scale(value, temperature, scale_by_temperature_squared)


def bucketed_reverse_kl(
    teacher_bucket_log_probs: torch.Tensor,
    student_bucket_log_probs: torch.Tensor,
    *,
    temperature: float = 1.0,
    scale_by_temperature_squared: bool = True,
) -> torch.Tensor:
    """``KL(student || teacher)`` over the ``top-k + tail`` partition."""
    value = kl_from_log_probs(student_bucket_log_probs, teacher_bucket_log_probs)
    return temperature_scale(value, temperature, scale_by_temperature_squared)


def bucketed_jsd(
    teacher_bucket_log_probs: torch.Tensor,
    student_bucket_log_probs: torch.Tensor,
    *,
    beta: float = 0.5,
    temperature: float = 1.0,
    scale_by_temperature_squared: bool = True,
) -> torch.Tensor:
    """Beta-weighted Jensen-Shannon divergence over the ``top-k + tail`` partition."""
    if not 0.0 < beta < 1.0:
        raise ValueError(
            f"jsd beta must be strictly inside (0, 1); got {beta}. "
            "At beta=0 or beta=1 the mixture equals one of the inputs and the "
            "divergence is identically zero."
        )
    log_beta = math.log(beta)
    log_one_minus_beta = math.log1p(-beta)
    log_mixture = torch.logaddexp(
        teacher_bucket_log_probs + log_beta,
        student_bucket_log_probs + log_one_minus_beta,
    )
    value = beta * kl_from_log_probs(teacher_bucket_log_probs, log_mixture) + (
        1.0 - beta
    ) * kl_from_log_probs(student_bucket_log_probs, log_mixture)
    return temperature_scale(value, temperature, scale_by_temperature_squared)


def bucketed_divergence(
    *,
    teacher_topk_log_probs: torch.Tensor,
    teacher_tail_log_prob: torch.Tensor,
    topk_indices: torch.Tensor,
    student_logits: torch.Tensor,
    divergence: str,
    temperature: float = 1.0,
    scale_by_temperature_squared: bool = True,
    jsd_beta: float = 0.5,
    tail_epsilon: float = 1e-9,
) -> torch.Tensor:
    """End-to-end bucketed divergence from cached teacher targets, shape ``[N]``.

    ``teacher_topk_log_probs`` / ``teacher_tail_log_prob`` are assumed to have
    been produced at the same ``temperature``; the caller (the trainer) verifies
    that against the cache index before getting here.
    """
    student_topk, student_tail = student_bucket_log_probs(
        student_logits, topk_indices, temperature=temperature
    )
    teacher_buckets, student_buckets = build_bucket_distributions(
        teacher_topk_log_probs,
        teacher_tail_log_prob,
        student_topk,
        student_tail,
        tail_epsilon=tail_epsilon,
    )
    if divergence == "forward_kl":
        return bucketed_forward_kl(
            teacher_buckets,
            student_buckets,
            temperature=temperature,
            scale_by_temperature_squared=scale_by_temperature_squared,
        )
    if divergence == "reverse_kl":
        return bucketed_reverse_kl(
            teacher_buckets,
            student_buckets,
            temperature=temperature,
            scale_by_temperature_squared=scale_by_temperature_squared,
        )
    if divergence == "jsd":
        return bucketed_jsd(
            teacher_buckets,
            student_buckets,
            beta=jsd_beta,
            temperature=temperature,
            scale_by_temperature_squared=scale_by_temperature_squared,
        )
    raise ValueError(f"unknown divergence {divergence!r}; expected forward_kl, reverse_kl or jsd")


def bucketed_teacher_entropy(
    teacher_topk_log_probs: torch.Tensor,
    teacher_tail_log_prob: torch.Tensor,
    *,
    tail_epsilon: float = 1e-9,
) -> torch.Tensor:
    """Coarse-grained teacher entropy over ``top-k + tail`` buckets, in nats.

    This is the entropy *of the bucketed distribution*, which lower-bounds the
    true full-vocabulary entropy because merging the tail into one bucket
    discards its internal spread.  Reports label it accordingly.
    """
    log_eps = math.log(tail_epsilon)
    buckets = torch.cat(
        [
            to_float32(teacher_topk_log_probs),
            to_float32(teacher_tail_log_prob).clamp_min(log_eps).unsqueeze(-1),
        ],
        dim=-1,
    )
    buckets = buckets - torch.logsumexp(buckets, dim=-1, keepdim=True)
    return entropy_from_log_probs(buckets)
