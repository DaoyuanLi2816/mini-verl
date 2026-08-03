"""Typed contracts for post-SFT alignment workflows and diagnostics."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "AlignmentMethod",
    "TeacherMode",
    "GateSignal",
    "ArtifactIdentity",
    "GateConfig",
    "DPOProvenance",
    "PilotEvidence",
    "PilotRecommendation",
    "PilotResult",
    "StateSource",
    "Supervision",
    "StateSupervisionRecord",
    "AlignmentMetrics",
    "AlignmentConfig",
    "BenchmarkAdapter",
    "BenchmarkRegistry",
]


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)


class AlignmentMethod(str, Enum):
    """Methods compared by the one-GPU alignment workflow."""

    SFT_CHECKPOINT = "sft_checkpoint"
    CONTINUED_SFT = "continued_sft"
    DPO = "dpo"
    OFFLINE_DISTILLATION = "offline_distillation"
    STANDARD_OPD = "standard_opd"
    VERIFIER_GATED_OPD = "verifier_gated_opd"


class TeacherMode(str, Enum):
    """Where the alignment target policy comes from."""

    POLICY_CONDITIONED = "policy_conditioned"
    ALIGNED_ADAPTER = "aligned_adapter"


class GateSignal(str, Enum):
    """Versioned qualification signals supported by v0.5."""

    POLICY_CRITICAL_SPAN = "policy_critical_span"


class StateSource(str, Enum):
    ORACLE = "oracle"
    FROZEN_STUDENT = "frozen_student"
    FRESH_STUDENT = "fresh_student"


class Supervision(str, Enum):
    HARD_TARGET = "hard_target"
    TEACHER_ARGMAX = "teacher_argmax"
    TEACHER_SOFT = "teacher_soft_distribution"
    PREFERENCE_REWARD = "preference_reward"


class PilotRecommendation(str, Enum):
    CONTINUED_SFT = "continued_sft"
    DPO = "dpo"
    OFFLINE_DISTILLATION = "offline_distillation"
    STANDARD_OPD = "standard_opd"
    VERIFIER_GATED_OPD = "verifier_gated_opd"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class ArtifactIdentity(_Base):
    """Immutable identity without a machine-local path."""

    id: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    license: str | None = None


class GateConfig(_Base):
    """Frozen calibration contract for Verifier-Gated OPD."""

    version: str = Field(min_length=1)
    signal: GateSignal
    decision_scope: Literal["span"] = "span"
    threshold: float = Field(default=1.0, ge=0.0, le=1.0)
    calibrated_on: Literal["train", "eval"] = "eval"
    frozen_before_test: bool = True

    @model_validator(mode="after")
    def _test_is_never_calibration(self) -> GateConfig:
        if not self.frozen_before_test:
            raise ValueError("gate.frozen_before_test must be true for a final-test run")
        return self


class DPOProvenance(_Base):
    """Import contract for an adapter trained by a pinned TRL DPO job."""

    trl_version: str = Field(min_length=1)
    exact_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reference_model: ArtifactIdentity
    dataset: ArtifactIdentity
    checkpoint: ArtifactIdentity
    adapter: ArtifactIdentity


class PilotEvidence(_Base):
    """Bounded diagnostic evidence; values are rates or signed rate gaps."""

    sample_size: int = Field(default=0, ge=0)
    teacher_policy_competence: float | None = Field(default=None, ge=0.0, le=1.0)
    student_baseline_alignment: float = Field(default=0.0, ge=0.0, le=1.0)
    teacher_student_policy_gap: float | None = Field(default=None, ge=-1.0, le=1.0)
    teacher_student_topk_overlap: float | None = Field(default=None, ge=0.0, le=1.0)
    fresh_state_gap: float = Field(default=0.0, ge=-1.0, le=1.0)
    hard_soft_gap: float = Field(default=0.0, ge=-1.0, le=1.0)
    preference_win_gap: float = Field(default=0.0, ge=-1.0, le=1.0)
    policy_sensitive_token_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    verifier_precision: float | None = Field(default=None, ge=0.0, le=1.0)
    estimated_vram_gib: float | None = Field(default=None, ge=0.0)
    estimated_time_seconds: float | None = Field(default=None, ge=0.0)
    uncertainty_half_width: float | None = Field(default=None, ge=0.0, le=1.0)


class PilotResult(_Base):
    schema_version: Literal[1] = 1
    rules_version: Literal["alignment-pilot-v1"] = "alignment-pilot-v1"
    recommendation: PilotRecommendation
    evidence: PilotEvidence
    reasons: list[str] = Field(min_length=1)
    cost_assumptions: dict[str, Any]
    uncertainty_note: str


class StateSupervisionRecord(_Base):
    state_source: StateSource
    supervision: Supervision
    state_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    teacher_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    starting_checkpoint_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    budget_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    seeds: list[int] = Field(min_length=1)
    outcome: dict[str, float] = Field(default_factory=dict)


class AlignmentMetrics(_Base):
    """Joint policy, utility and cost endpoints; no opaque aggregate."""

    tasks: int = Field(default=0, ge=0)
    alignment_score: float = Field(default=0.0, ge=0.0, le=1.0)
    appropriate_refusal_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    harmful_compliance_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    over_refusal_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    benign_compliance_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    preference_win_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    instruction_retention: float | None = Field(default=None, ge=0.0, le=1.0)
    tool_utility_retention: float | None = Field(default=None, ge=0.0, le=1.0)
    general_utility_retention: float | None = Field(default=None, ge=0.0, le=1.0)
    teacher_queried_positions: int | None = Field(default=None, ge=0)
    teacher_query_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    gpu_seconds: float | None = Field(default=None, ge=0.0)
    peak_vram_bytes: int | None = Field(default=None, ge=0)
    decision_distribution_shift_jsd: float | None = Field(default=None, ge=0.0, le=1.0)


class AlignmentConfig(_Base):
    """Optional top-level section that turns a run into an alignment workflow."""

    schema_version: Literal[1] = 1
    method: AlignmentMethod
    teacher_mode: TeacherMode | None = None
    starting_sft_checkpoint: str | None = None
    starting_sft_checkpoint_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    policy: ArtifactIdentity
    reference: ArtifactIdentity | None = None
    gate: GateConfig | None = None
    dpo: DPOProvenance | None = None
    dpo_adapter_path: str | None = None
    evaluation_adapters: list[str] = Field(
        default_factory=lambda: ["minipolicy_v1", "ifeval", "xstest", "harmbench", "rewardbench"]
    )
    pilot: PilotEvidence | None = None
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _method_contract(self) -> AlignmentConfig:
        if bool(self.starting_sft_checkpoint) != bool(self.starting_sft_checkpoint_sha256):
            raise ValueError(
                "starting_sft_checkpoint and starting_sft_checkpoint_sha256 must be provided together"
            )
        if self.method is AlignmentMethod.VERIFIER_GATED_OPD and self.gate is None:
            raise ValueError("verifier_gated_opd requires an explicit frozen gate")
        if self.method is not AlignmentMethod.VERIFIER_GATED_OPD and self.gate is not None:
            raise ValueError("alignment.gate applies only to verifier_gated_opd")
        if self.method is AlignmentMethod.DPO and self.dpo is None:
            raise ValueError(
                "dpo requires pinned TRL, dataset, reference, checkpoint and adapter provenance"
            )
        if self.method is AlignmentMethod.DPO and not self.dpo_adapter_path:
            raise ValueError("dpo requires a local dpo_adapter_path for deterministic evaluation")
        if self.method is not AlignmentMethod.DPO and self.dpo is not None:
            raise ValueError("alignment.dpo provenance applies only to method=dpo")
        if self.method is not AlignmentMethod.DPO and self.dpo_adapter_path is not None:
            raise ValueError("alignment.dpo_adapter_path applies only to method=dpo")
        return self


class BenchmarkAdapter(_Base):
    name: str
    dimensions: list[str] = Field(min_length=1)
    repository: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    license: str
    redistribute_data: bool
    integration: Literal["metadata_only", "external_responses"]
    notes: str


class BenchmarkRegistry(_Base):
    schema_version: Literal[1] = 1
    audited_at: str
    benchmarks: list[BenchmarkAdapter] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_names(self) -> BenchmarkRegistry:
        names = [item.name for item in self.benchmarks]
        if len(names) != len(set(names)):
            raise ValueError("benchmark adapter names must be unique")
        return self
