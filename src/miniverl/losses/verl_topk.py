"""Pinned verl v0.8 ``forward_kl_topk`` semantics.

This objective intentionally ignores the probability tail.  It is therefore
separate from miniVERL's native ``bucketed_topk_tail`` coarse-grained KL.  The
implementation follows official verl v0.8.0 at commit
``7aed6b230776f963fa09509c10d9c3a767d1102c``; the optional conformance gate
loads that source directly and compares values and gradients.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

__all__ = ["VERL_TOPK_SCORE_IMPLEMENTATION", "VerlTopKOutput", "verl_forward_kl_topk"]

VERL_TOPK_SCORE_IMPLEMENTATION = "verl-v0.8.0-forward-kl-topk-v1"


@dataclass(frozen=True)
class VerlTopKOutput:
    """Per-position loss and official OPD diagnostics."""

    loss: torch.Tensor
    student_mass: torch.Tensor
    teacher_mass: torch.Tensor
    overlap_count: torch.Tensor
    overlap_token_advantage: torch.Tensor


def verl_forward_kl_topk(
    student_logits: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    *,
    log_prob_min_clamp: float | None = None,
    loss_max_clamp: float | None = None,
) -> VerlTopKOutput:
    """Compute the supported direct-supervision verl v0.8 top-k objective.

    Mass and student top-k identities are derived before log-prob clamps.  Both
    gathered student and teacher log-probabilities are then minimum-clamped,
    the unnormalised teacher top-k terms are summed, and the per-token value is
    clamped non-negative.  The optional symmetric maximum clamp is the later
    official distillation-wrapper stage.
    """
    if student_logits.ndim < 2:
        raise ValueError("student_logits must have shape [..., vocab]")
    expected = (*student_logits.shape[:-1], teacher_topk_ids.shape[-1])
    if tuple(teacher_topk_ids.shape) != expected:
        raise ValueError(
            f"teacher_topk_ids shape {tuple(teacher_topk_ids.shape)} does not match {expected}"
        )
    if teacher_topk_log_probs.shape != teacher_topk_ids.shape:
        raise ValueError("teacher top-k IDs and log-probabilities must have identical shapes")
    if teacher_topk_ids.dtype not in {torch.int32, torch.int64}:
        raise ValueError("teacher_topk_ids must be an integer tensor")
    if log_prob_min_clamp is not None and not torch.isfinite(torch.tensor(log_prob_min_clamp)):
        raise ValueError("log_prob_min_clamp must be finite or None")
    if loss_max_clamp is not None and (
        loss_max_clamp <= 0 or not torch.isfinite(torch.tensor(loss_max_clamp))
    ):
        raise ValueError("loss_max_clamp must be positive and finite or None")

    student_log_probs = F.log_softmax(student_logits, dim=-1)
    k = teacher_topk_ids.shape[-1]
    student_topk_ids = torch.topk(student_log_probs, k=k, dim=-1).indices
    student_selected = torch.gather(student_log_probs, dim=-1, index=teacher_topk_ids)
    student_mass = student_selected.exp().sum(dim=-1)
    teacher_mass = teacher_topk_log_probs.exp().sum(dim=-1)

    teacher_used = teacher_topk_log_probs.float()
    student_used = student_selected.float()
    if log_prob_min_clamp is not None:
        teacher_used = teacher_used.clamp_min(log_prob_min_clamp)
        student_used = student_used.clamp_min(log_prob_min_clamp)
    per_teacher_token = teacher_used.exp() * (teacher_used - student_used)
    loss = per_teacher_token.sum(dim=-1).clamp_min(0.0)
    if loss_max_clamp is not None:
        loss = loss.clamp(max=loss_max_clamp)

    overlap_mask = (teacher_topk_ids.unsqueeze(-1) == student_topk_ids.unsqueeze(-2)).any(dim=-1)
    overlap_count = overlap_mask.sum(dim=-1)
    advantage_sum = (-per_teacher_token * overlap_mask).sum(dim=-1)
    overlap_token_advantage = advantage_sum / overlap_count.clamp_min(1)
    overlap_token_advantage = torch.where(
        overlap_count > 0,
        overlap_token_advantage,
        torch.zeros_like(overlap_token_advantage),
    )
    return VerlTopKOutput(
        loss=loss,
        student_mass=student_mass,
        teacher_mass=teacher_mass,
        overlap_count=overlap_count,
        overlap_token_advantage=overlap_token_advantage,
    )
