"""Chunked selected-position objective.

Why this module exists
----------------------
A naive distillation step computes ``[batch, seq_len, vocab]`` logits.  For a
152k-vocabulary model at sequence length 768 that is 116 M floats per sequence
*before* the backward pass -- far beyond a 16 GB card.  miniVERL never builds
that tensor.  Instead:

1. The backbone runs once and produces hidden states.
2. Only the **selected prediction positions** are gathered: ``[N, H]``.
3. Those are projected through the LM head in chunks of ``chunk_size``, so the
   largest vocabulary-sized tensor alive at any moment is ``[chunk_size, V]``.

Getting the gradient right
--------------------------
Backpropagating each chunk straight through the backbone would re-run the
backbone once per chunk.  Instead the selected hidden states are *detached*
into a leaf ``work`` tensor, each chunk's loss is backpropagated into
``work.grad`` (freeing its ``[chunk, V]`` intermediates immediately), and a
single ``hidden_states.backward(gradient=work.grad)`` at the end pushes the
accumulated gradient through the backbone exactly once.

Because every chunk divides by the *global* weight sum, the sum of the chunk
losses equals the unchunked loss, and ``work.grad`` equals the unchunked
gradient.  ``tests/unit/test_chunked_equivalence.py`` asserts both.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import torch

from miniverl.losses.bucketed import (
    bucketed_divergence,
    bucketed_teacher_entropy,
)
from miniverl.losses.exact import exact_divergence, exact_teacher_entropy
from miniverl.losses.reduction import MIN_TOTAL_WEIGHT

__all__ = [
    "ChunkTargetProvider",
    "ExactTargetProvider",
    "BucketedTargetProvider",
    "LossOutput",
    "chunked_selected_position_loss",
]


class ChunkTargetProvider(Protocol):
    """Supplies teacher supervision for a slice of selected positions."""

    kind: str

    def divergence(self, start: int, end: int, student_logits: torch.Tensor) -> torch.Tensor:
        """Per-position divergence ``[end - start]`` for this chunk."""
        ...

    def teacher_entropy(self, start: int, end: int) -> torch.Tensor:
        """Per-position teacher entropy in nats ``[end - start]`` (no grad)."""
        ...


@dataclass
class ExactTargetProvider:
    """Full-vocabulary teacher supervision, computed lazily per chunk.

    ``teacher_logits_fn(start, end)`` returns ``[end - start, V]`` teacher
    logits.  Passing a callable rather than a materialized tensor is what keeps
    the teacher side to one chunk of vocabulary-sized memory at a time.
    """

    teacher_logits_fn: Callable[[int, int], torch.Tensor]
    divergence_name: str = "reverse_kl"
    temperature: float = 1.0
    scale_by_temperature_squared: bool = True
    jsd_beta: float = 0.5
    kind: str = "exact"

    def divergence(self, start: int, end: int, student_logits: torch.Tensor) -> torch.Tensor:
        teacher_logits = self.teacher_logits_fn(start, end)
        return exact_divergence(
            teacher_logits,
            student_logits,
            divergence=self.divergence_name,
            temperature=self.temperature,
            scale_by_temperature_squared=self.scale_by_temperature_squared,
            jsd_beta=self.jsd_beta,
        )

    def teacher_entropy(self, start: int, end: int) -> torch.Tensor:
        with torch.no_grad():
            return exact_teacher_entropy(
                self.teacher_logits_fn(start, end), temperature=self.temperature
            )


@dataclass
class BucketedTargetProvider:
    """``top-k + tail`` teacher supervision from live scoring or the cache."""

    topk_indices: torch.Tensor
    topk_log_probs: torch.Tensor
    tail_log_prob: torch.Tensor
    divergence_name: str = "reverse_kl"
    temperature: float = 1.0
    scale_by_temperature_squared: bool = True
    jsd_beta: float = 0.5
    tail_epsilon: float = 1e-9
    kind: str = "bucketed"

    def divergence(self, start: int, end: int, student_logits: torch.Tensor) -> torch.Tensor:
        return bucketed_divergence(
            teacher_topk_log_probs=self.topk_log_probs[start:end],
            teacher_tail_log_prob=self.tail_log_prob[start:end],
            topk_indices=self.topk_indices[start:end],
            student_logits=student_logits,
            divergence=self.divergence_name,
            temperature=self.temperature,
            scale_by_temperature_squared=self.scale_by_temperature_squared,
            jsd_beta=self.jsd_beta,
            tail_epsilon=self.tail_epsilon,
        )

    def teacher_entropy(self, start: int, end: int) -> torch.Tensor:
        with torch.no_grad():
            return bucketed_teacher_entropy(
                self.topk_log_probs[start:end],
                self.tail_log_prob[start:end],
                tail_epsilon=self.tail_epsilon,
            )


@dataclass
class LossOutput:
    """Result of one chunked objective evaluation."""

    loss: float
    num_positions: int
    total_weight: float
    per_token: torch.Tensor
    per_token_ce: torch.Tensor | None = None
    teacher_entropy: torch.Tensor | None = None
    grad_hidden: torch.Tensor | None = None
    num_chunks: int = 0
    metrics: dict[str, float] = field(default_factory=dict)


def _cross_entropy(student_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Per-position next-token cross-entropy in nats, float32."""
    log_probs = torch.log_softmax(student_logits.to(torch.float32), dim=-1)
    return -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def chunked_selected_position_loss(
    *,
    hidden_states: torch.Tensor,
    lm_head: Callable[[torch.Tensor], torch.Tensor],
    weights: torch.Tensor,
    provider: ChunkTargetProvider | None = None,
    target_token_ids: torch.Tensor | None = None,
    ce_weight: float = 0.0,
    chunk_size: int = 256,
    backward: bool = False,
    loss_scale: float = 1.0,
    collect_teacher_entropy: bool = False,
) -> LossOutput:
    """Evaluate (and optionally backpropagate) the objective in vocabulary chunks.

    Parameters
    ----------
    hidden_states:
        ``[N, H]`` student hidden states at the **selected prediction
        positions** only.
    lm_head:
        Callable mapping ``[c, H] -> [c, V]``.
    weights:
        ``[N]`` non-negative token weights.  A zero weight contributes exactly
        zero to both the loss and the gradient.
    provider:
        Teacher supervision.  ``None`` means pure cross-entropy (SFT).
    target_token_ids:
        ``[N]`` target ids; required when ``ce_weight > 0`` or ``provider`` is
        ``None``.
    ce_weight:
        Convex mixing weight of the cross-entropy term:
        ``loss = (1 - ce_weight) * divergence + ce_weight * ce``.
    chunk_size:
        Selected positions projected through the LM head at once.  Purely a
        memory knob; it does not change the mathematical objective.
    backward:
        When true, gradients are accumulated as the chunks are processed and a
        single backward through the backbone is issued at the end.
    loss_scale:
        Multiplier applied before ``backward`` (gradient accumulation).  The
        returned ``loss`` is the *unscaled* value.
    """
    if provider is None and ce_weight <= 0.0:
        raise ValueError("either a teacher target provider or ce_weight > 0 is required")
    if (ce_weight > 0.0 or provider is None) and target_token_ids is None:
        raise ValueError("target_token_ids are required for the cross-entropy term")
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")

    n = int(hidden_states.shape[0])
    device = hidden_states.device
    if n == 0:
        zero = torch.zeros((), dtype=torch.float32, device=device)
        return LossOutput(
            loss=0.0,
            num_positions=0,
            total_weight=0.0,
            per_token=torch.zeros(0, dtype=torch.float32, device=device),
            per_token_ce=torch.zeros(0, dtype=torch.float32, device=device),
            teacher_entropy=torch.zeros(0, dtype=torch.float32, device=device)
            if collect_teacher_entropy
            else None,
            grad_hidden=None,
            num_chunks=0,
            metrics={"empty_batch": 1.0, "loss_value": float(zero)},
        )

    w = weights.to(torch.float32).to(device)
    if w.shape[0] != n:
        raise ValueError(f"weights has length {w.shape[0]} but hidden_states has {n} rows")
    denom = torch.clamp(w.sum(), min=MIN_TOTAL_WEIGHT)

    use_two_stage = backward and hidden_states.requires_grad
    work = hidden_states.detach().requires_grad_(True) if use_two_stage else hidden_states

    loss_value = 0.0
    per_token_parts: list[torch.Tensor] = []
    ce_parts: list[torch.Tensor] = []
    entropy_parts: list[torch.Tensor] = []
    num_chunks = 0

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        num_chunks += 1
        chunk_hidden = work[start:end]
        student_logits = lm_head(chunk_hidden)

        if provider is not None:
            divergence = provider.divergence(start, end, student_logits)
        else:
            divergence = torch.zeros(end - start, dtype=torch.float32, device=device)

        if ce_weight > 0.0 or provider is None:
            assert target_token_ids is not None  # narrowed by the guard above
            ce = _cross_entropy(student_logits, target_token_ids[start:end].to(device))
        else:
            ce = torch.zeros(end - start, dtype=torch.float32, device=device)

        if provider is None:
            combined = ce
        elif ce_weight > 0.0:
            combined = (1.0 - ce_weight) * divergence + ce_weight * ce
        else:
            combined = divergence

        chunk_loss = (combined * w[start:end]).sum() / denom
        if backward:
            (chunk_loss * loss_scale).backward()
        loss_value += float(chunk_loss.detach())

        per_token_parts.append(divergence.detach())
        ce_parts.append(ce.detach())
        if collect_teacher_entropy and provider is not None:
            entropy_parts.append(provider.teacher_entropy(start, end).detach())

        del student_logits, divergence, ce, combined, chunk_loss

    grad_hidden: torch.Tensor | None = None
    if use_two_stage:
        grad_hidden = work.grad
        if grad_hidden is None:
            grad_hidden = torch.zeros_like(work)
        hidden_states.backward(gradient=grad_hidden)

    return LossOutput(
        loss=loss_value,
        num_positions=n,
        total_weight=float(w.sum()),
        per_token=torch.cat(per_token_parts) if per_token_parts else torch.zeros(0),
        per_token_ce=torch.cat(ce_parts) if ce_parts else None,
        teacher_entropy=torch.cat(entropy_parts) if entropy_parts else None,
        grad_hidden=grad_hidden,
        num_chunks=num_chunks,
        metrics={
            "loss_value": loss_value,
            "num_positions": float(n),
            "total_weight": float(w.sum()),
            "num_chunks": float(num_chunks),
        },
    )
