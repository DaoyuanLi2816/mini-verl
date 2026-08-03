"""One-GPU post-SFT alignment workflow, diagnostics and cards."""

from __future__ import annotations

from miniverl.alignment.card import render_alignment_card
from miniverl.alignment.evaluation import alignment_metrics, validate_state_supervision_matrix
from miniverl.alignment.pilot import recommend_alignment_method
from miniverl.alignment.preferences import (
    build_tool_policy_preferences,
    preference_dataset_digest,
)
from miniverl.alignment.registry import load_benchmark_registry
from miniverl.alignment.schema import (
    AlignmentMethod,
    AlignmentMetrics,
    ArtifactIdentity,
    BenchmarkAdapter,
    BenchmarkRegistry,
    PilotEvidence,
    PilotRecommendation,
    PilotResult,
    StateSource,
    StateSupervisionRecord,
    Supervision,
)
from miniverl.alignment.workflow import (
    build_alignment_stage_plan,
    load_alignment_method_adapter,
    load_alignment_starting_checkpoint,
    publish_alignment_artifacts,
    run_alignment,
)

__all__ = [
    "AlignmentMethod",
    "AlignmentMetrics",
    "ArtifactIdentity",
    "BenchmarkAdapter",
    "BenchmarkRegistry",
    "PilotEvidence",
    "PilotRecommendation",
    "PilotResult",
    "StateSource",
    "StateSupervisionRecord",
    "Supervision",
    "alignment_metrics",
    "build_alignment_stage_plan",
    "build_tool_policy_preferences",
    "load_benchmark_registry",
    "recommend_alignment_method",
    "render_alignment_card",
    "load_alignment_starting_checkpoint",
    "load_alignment_method_adapter",
    "publish_alignment_artifacts",
    "preference_dataset_digest",
    "run_alignment",
    "validate_state_supervision_matrix",
]
