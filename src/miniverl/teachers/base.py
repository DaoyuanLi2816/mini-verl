"""Teacher-scoring contract.

A scorer is handed the *student's own* trajectory and an alignment map, and
returns supervision for exactly the aligned positions.  It never re-generates,
re-tokenizes or "improves" the trajectory: that would silently turn on-policy
distillation into off-policy distillation on teacher text.

Two supervision shapes are produced, both of which
:mod:`miniverl.losses.chunked` consumes through the same
:class:`~miniverl.losses.chunked.ChunkTargetProvider` interface:

``exact_hidden``
    Teacher hidden states at the aligned positions plus the teacher's LM head.
    The full ``[chunk, V]`` distribution is rebuilt one chunk at a time and
    thrown away immediately.  Memory is ``[N, H] + [chunk, V]``, so exact
    full-vocabulary KL is affordable even for a 152k vocabulary -- but the
    teacher must stay resident, because its LM head is needed during the update.

``bucketed``
    Compressed ``top-k + tail`` targets.  Serializable, cacheable, and the only
    shape that survives the teacher being evicted from VRAM.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from miniverl.schemas.alignment import AlignmentMap
from miniverl.schemas.cache import TeacherTargetBatch
from miniverl.schemas.trajectory import Trajectory

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

    from miniverl.losses.chunked import ChunkTargetProvider

__all__ = ["TeacherScoreResult", "TeacherScorer"]


@dataclass
class TeacherScoreResult:
    """Supervision for one trajectory, ready for the loss."""

    trajectory_id: str
    policy_version: int
    shape: str
    provider: ChunkTargetProvider
    target_token_ids: torch.Tensor
    weights: torch.Tensor
    span_types: list[str]
    teacher_entropy: torch.Tensor
    num_positions: int
    #: Present only when the supervision can be persisted (``bucketed`` shape).
    cacheable: TeacherTargetBatch | None = None
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def is_cacheable(self) -> bool:
        """``True`` when this result can be written to the teacher cache."""
        return self.cacheable is not None


class TeacherScorer(ABC):
    """Produces token-level supervision on states the student actually visited."""

    @abstractmethod
    def score(
        self,
        *,
        student: Trajectory,
        alignment: AlignmentMap,
        teacher_view: Trajectory | None = None,
    ) -> TeacherScoreResult:
        """Score ``alignment``'s teacher positions.

        Parameters
        ----------
        student:
            The trajectory the policy produced.
        alignment:
            Aligned student/teacher prediction positions and token weights.
        teacher_view:
            The privileged re-render, or ``None`` for the standard mode in which
            the teacher reads the student's own token sequence.
        """
        ...

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        """Identity of the teacher, recorded in the run manifest and the cache."""
        ...
