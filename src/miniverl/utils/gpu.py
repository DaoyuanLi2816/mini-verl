"""CUDA memory accounting.

``torch.cuda.max_memory_allocated`` and ``max_memory_reserved`` are both
reported, because they answer different questions: *allocated* is what the
tensors need, *reserved* is what the caching allocator took from the driver and
therefore what actually decides whether the next run OOMs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from miniverl.utils.lazy import have_module

__all__ = ["MemorySnapshot", "cuda_available", "reset_peak_stats", "snapshot", "empty_cache"]


def cuda_available() -> bool:
    """``True`` when a CUDA device is usable."""
    if not have_module("torch"):
        return False
    import torch

    return bool(torch.cuda.is_available())


@dataclass(frozen=True)
class MemorySnapshot:
    """Peak and current CUDA memory in bytes."""

    available: bool
    allocated_bytes: int = 0
    reserved_bytes: int = 0
    peak_allocated_bytes: int = 0
    peak_reserved_bytes: int = 0
    total_bytes: int = 0

    @property
    def peak_allocated_gib(self) -> float:
        """Peak allocated memory in GiB."""
        return self.peak_allocated_bytes / (1024**3)

    @property
    def peak_reserved_gib(self) -> float:
        """Peak reserved memory in GiB."""
        return self.peak_reserved_bytes / (1024**3)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly view for metrics and reports."""
        return {
            "cuda_available": self.available,
            "allocated_bytes": self.allocated_bytes,
            "reserved_bytes": self.reserved_bytes,
            "peak_allocated_bytes": self.peak_allocated_bytes,
            "peak_reserved_bytes": self.peak_reserved_bytes,
            "peak_allocated_gib": round(self.peak_allocated_gib, 4),
            "peak_reserved_gib": round(self.peak_reserved_gib, 4),
            "total_bytes": self.total_bytes,
        }


def reset_peak_stats() -> None:
    """Reset CUDA peak counters so the next phase is measured on its own."""
    if not cuda_available():
        return
    import torch

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()


def snapshot() -> MemorySnapshot:
    """Read the current and peak CUDA memory counters."""
    if not cuda_available():
        return MemorySnapshot(available=False)
    import torch

    torch.cuda.synchronize()
    total_bytes = torch.cuda.mem_get_info()[1]
    return MemorySnapshot(
        available=True,
        allocated_bytes=int(torch.cuda.memory_allocated()),
        reserved_bytes=int(torch.cuda.memory_reserved()),
        peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
        peak_reserved_bytes=int(torch.cuda.max_memory_reserved()),
        total_bytes=int(total_bytes),
    )


def free_vram_gib() -> float:
    """Free VRAM in GiB, or ``0.0`` without CUDA."""
    if not cuda_available():
        return 0.0
    import torch

    return float(torch.cuda.mem_get_info()[0]) / (1024**3)


def empty_cache() -> None:
    """Return cached blocks to the driver."""
    if not cuda_available():
        return
    import gc

    import torch

    gc.collect()
    torch.cuda.empty_cache()


def is_oom_error(exc: BaseException) -> bool:
    """``True`` for CUDA out-of-memory errors, including the CPU allocator's."""
    if have_module("torch"):
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    message = str(exc).lower()
    return "out of memory" in message or "cuda error: out of memory" in message
