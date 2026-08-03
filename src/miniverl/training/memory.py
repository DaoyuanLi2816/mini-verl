"""Consumer-GPU memory strategies.

``resident``
    Teacher and student both stay on the accelerator.  Cheapest in wall-clock
    and the only strategy that supports the ``exact_hidden`` teacher shape,
    because the teacher's LM head must be callable during the update.

``swap``
    One model on the accelerator at a time.  Per cycle: student rolls out ->
    student weights *and optimizer state* move to host memory and the student's
    device memory is released -> teacher scores and writes compressed targets ->
    teacher is released -> student and optimizer state come back -> update runs
    against the cached targets.  Slower, but it fits pairs that ``resident``
    cannot.

``auto``
    Tries ``resident`` and falls back to ``swap`` when the teacher does not fit.
    The decision and its reason are printed, written to
    ``config.resolved.yaml`` and recorded in the manifest -- ``auto`` never
    silently changes anything that affects the mathematical objective.

OOM handling only ever halves ``loss.chunk_size``.  That is a pure
memory/throughput knob: the loss and the gradient are identical for any chunk
size (asserted in ``tests/unit/test_chunked_equivalence.py``).  Sequence
lengths, batch sizes, models and objectives are never changed behind the user's
back; if halving the chunk down to ``memory.min_chunk_size`` still OOMs, the run
fails with concrete advice.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

from miniverl.config.models import MemoryConfig, MemoryStrategy
from miniverl.errors import GpuMemoryError
from miniverl.utils.gpu import empty_cache, free_vram_gib, is_oom_error

__all__ = ["MemoryPlan", "resolve_strategy", "run_with_oom_retry", "move_optimizer_state"]

T = TypeVar("T")


@dataclass
class MemoryPlan:
    """The resolved memory decision for a run."""

    strategy: MemoryStrategy
    chunk_size: int
    device: str
    reason: str
    oom_retries_used: int = 0
    chunk_size_history: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly view for the manifest."""
        return {
            "strategy": self.strategy.value,
            "projection_chunk_size": self.chunk_size,
            "device": self.device,
            "reason": self.reason,
            "oom_retries_used": self.oom_retries_used,
            "chunk_size_history": list(self.chunk_size_history),
        }


def resolve_strategy(
    memory: MemoryConfig,
    *,
    device: str,
    chunk_size: int,
    teacher_fits: Callable[[], bool] | None = None,
) -> MemoryPlan:
    """Turn ``auto`` into a concrete strategy and explain the choice.

    ``teacher_fits`` is called only for ``auto`` on CUDA.  It should attempt the
    resident placement and return ``False`` (having cleaned up) if it does not
    fit.
    """
    if memory.strategy is not MemoryStrategy.AUTO:
        return MemoryPlan(
            strategy=memory.strategy,
            chunk_size=chunk_size,
            device=device,
            reason=f"memory.strategy was set explicitly to {memory.strategy.value}",
        )

    if not device.startswith("cuda"):
        return MemoryPlan(
            strategy=MemoryStrategy.RESIDENT,
            chunk_size=chunk_size,
            device=device,
            reason="auto -> resident: no CUDA device, host memory is not partitioned",
        )

    free_gib = free_vram_gib()
    if free_gib < memory.auto_swap_vram_headroom_gb:
        return MemoryPlan(
            strategy=MemoryStrategy.SWAP,
            chunk_size=chunk_size,
            device=device,
            reason=(
                f"auto -> swap: only {free_gib:.2f} GiB free before loading the teacher, "
                f"below the {memory.auto_swap_vram_headroom_gb:.2f} GiB headroom"
            ),
        )

    if teacher_fits is not None and not teacher_fits():
        return MemoryPlan(
            strategy=MemoryStrategy.SWAP,
            chunk_size=chunk_size,
            device=device,
            reason=(
                f"auto -> swap: the teacher did not fit alongside the student "
                f"({free_gib:.2f} GiB were free)"
            ),
        )

    return MemoryPlan(
        strategy=MemoryStrategy.RESIDENT,
        chunk_size=chunk_size,
        device=device,
        reason=f"auto -> resident: {free_gib:.2f} GiB free after loading the student",
    )


def run_with_oom_retry(
    fn: Callable[[int], T],
    *,
    plan: MemoryPlan,
    memory: MemoryConfig,
    on_retry: Callable[[int, int], None] | None = None,
    cleanup: Callable[[], None] | None = None,
) -> T:
    """Run ``fn(chunk_size)``, halving the chunk on CUDA OOM.

    The chunk size that finally succeeded is written back into ``plan`` so the
    rest of the run keeps using it instead of re-discovering the limit on every
    step.
    """
    chunk = plan.chunk_size
    attempts = memory.oom_retries + 1
    last_error: BaseException | None = None
    for attempt in range(attempts):
        try:
            result = fn(chunk)
            if chunk != plan.chunk_size:
                plan.chunk_size_history.append(plan.chunk_size)
                plan.chunk_size = chunk
            return result
        except (RuntimeError, MemoryError) as exc:
            if not is_oom_error(exc):
                raise
            last_error = exc
            if cleanup is not None:
                cleanup()
            empty_cache()
            next_chunk = max(chunk // 2, memory.min_chunk_size)
            if next_chunk == chunk or attempt == attempts - 1:
                break
            if on_retry is not None:
                on_retry(chunk, next_chunk)
            plan.oom_retries_used += 1
            chunk = next_chunk

    raise GpuMemoryError(
        f"CUDA ran out of memory and the {memory.oom_retries} equivalence-preserving "
        f"retries were exhausted (projection chunk size reached {chunk}).",
        hint=(
            "reduce train.trajectory_batch_size, reduce rollout.max_total_tokens, "
            "reduce train.gradient_accumulation_steps, lower loss.top_k, switch "
            "models.student.quantization to nf4, enable "
            "models.student.gradient_checkpointing, or set memory.strategy: swap. "
            f"Original error: {last_error}"
        ),
    )


def move_optimizer_state(optimizer: Any, device: str) -> None:
    """Move an optimizer's tensor state to ``device``.

    Without this, ``swap`` would move the model to host memory while leaving the
    Adam moments pinned on the GPU -- which is most of what an optimizer costs.
    """
    import torch

    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)
