"""Pure, torch-free data schemas for trajectories, caches and metrics."""

from __future__ import annotations

from miniverl.schemas.alignment import AlignmentMap
from miniverl.schemas.cache import CacheEntryMeta, CacheIndex, TeacherTargetBatch
from miniverl.schemas.trajectory import (
    CRITICAL_SPAN_TYPES,
    MODEL_GENERATED_SPAN_TYPES,
    Span,
    SpanType,
    TerminationReason,
    ToolCallRecord,
    ToolResultRecord,
    Trajectory,
    Turn,
    VerificationRecord,
)

__all__ = [
    "AlignmentMap",
    "CacheEntryMeta",
    "CacheIndex",
    "TeacherTargetBatch",
    "CRITICAL_SPAN_TYPES",
    "MODEL_GENERATED_SPAN_TYPES",
    "Span",
    "SpanType",
    "TerminationReason",
    "ToolCallRecord",
    "ToolResultRecord",
    "Trajectory",
    "Turn",
    "VerificationRecord",
]
