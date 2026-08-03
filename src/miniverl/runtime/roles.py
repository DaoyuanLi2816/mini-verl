"""verl-style semantic roles implemented as a small local process graph.

These protocols name ownership boundaries; they deliberately do not reproduce
verl's distributed transport, Ray workers, DataProto, FSDP, or scheduler APIs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

__all__ = [
    "ActorPolicy",
    "RolloutRuntime",
    "TeacherPolicy",
    "ReferencePolicy",
    "RewardOrVerifier",
    "TargetBuilder",
    "UpdateRuntime",
    "EvaluationRuntime",
    "ArtifactBridge",
    "LocalArtifactBridge",
    "LocalRoleGraph",
]


class ActorPolicy(Protocol):
    """Policy that generates trajectories and owns trainable parameters."""

    def trainable_parameters(self) -> list[Any]: ...


class RolloutRuntime(Protocol):
    """Runtime that turns tasks into actor trajectories."""

    def rollout(self, *args: Any, **kwargs: Any) -> Any: ...


class TeacherPolicy(Protocol):
    """Frozen policy that constructs distillation targets."""

    def hidden_states_at(self, *args: Any, **kwargs: Any) -> Any: ...


class ReferencePolicy(Protocol):
    """Frozen policy used as a policy-regularization reference."""

    def hidden_states_at(self, *args: Any, **kwargs: Any) -> Any: ...


class RewardOrVerifier(Protocol):
    """Deterministic task reward or verifier boundary."""

    def verify(self, answer: str) -> Any: ...


class TargetBuilder(Protocol):
    """Builds typed teacher targets on actor-visited states."""

    def score(self, *args: Any, **kwargs: Any) -> Any: ...


class UpdateRuntime(Protocol):
    """Owns objective evaluation and optimizer commits."""

    def train(self) -> Any: ...


class EvaluationRuntime(Protocol):
    """Owns deterministic policy evaluation."""

    def evaluate(self, *args: Any, **kwargs: Any) -> Any: ...


class ArtifactBridge(Protocol):
    """Publishes local run artifacts for later inspection or export."""

    def describe(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class LocalArtifactBridge:
    """Filesystem artifact boundary for one local run."""

    run_root: Path

    def describe(self) -> dict[str, Any]:
        """Return manifest-ready artifact storage metadata."""
        return {"kind": "local_filesystem", "run_root": str(self.run_root.resolve())}


@dataclass(frozen=True)
class LocalRoleGraph:
    """Typed role wiring for miniVERL's single-process execution model."""

    actor_policy: ActorPolicy
    rollout_runtime: RolloutRuntime
    teacher_policy: TeacherPolicy | None
    reference_policy: ReferencePolicy | None
    reward_or_verifier: RewardOrVerifier
    target_builder: TargetBuilder | None
    update_runtime: UpdateRuntime
    evaluation_runtime: EvaluationRuntime
    artifact_bridge: ArtifactBridge

    def __post_init__(self) -> None:
        if (
            self.teacher_policy is not None
            and self.reference_policy is not None
            and self.teacher_policy is self.reference_policy
        ):
            raise ValueError("teacher and reference must be distinct role views")

    def describe(self) -> dict[str, Any]:
        """Return semantic role names without implying distributed parity."""

        def kind(value: object | None) -> str | None:
            return type(value).__name__ if value is not None else None

        return {
            "execution_model": "local_single_process",
            "roles": {
                "actor_policy": kind(self.actor_policy),
                "rollout_runtime": kind(self.rollout_runtime),
                "teacher_policy": kind(self.teacher_policy),
                "reference_policy": kind(self.reference_policy),
                "reward_or_verifier": kind(self.reward_or_verifier),
                "target_builder": kind(self.target_builder),
                "update_runtime": kind(self.update_runtime),
                "evaluation_runtime": kind(self.evaluation_runtime),
                "artifact_bridge": kind(self.artifact_bridge),
            },
        }
