"""Trajectory inspection.

Torch-free: ``miniverl inspect`` validates and summarizes a trajectory JSONL
file using nothing but the base install.  The provenance summary it prints is
computed from the span partition, which is the same source the training masks
come from -- so if the file claims tool output is trainable, this command says
so instead of hiding it.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from miniverl.schemas.trajectory import (
    CRITICAL_SPAN_TYPES,
    MODEL_GENERATED_SPAN_TYPES,
    Trajectory,
)
from miniverl.trajectory.io import iter_trajectories

__all__ = ["TrajectorySummary", "FileSummary", "summarize_file", "summarize_trajectory"]


@dataclass
class TrajectorySummary:
    """Per-trajectory inspection record."""

    trajectory_id: str
    task_id: str
    environment: str
    policy_version: int
    termination_reason: str
    solved: bool | None
    reward: float | None
    tokens: int
    model_tokens: int
    critical_tokens: int
    context_tokens: int
    turns: int
    valid_tool_calls: int
    invalid_tool_calls: int
    generated_tokens: int
    tokens_by_span_type: dict[str, int] = field(default_factory=dict)
    tools_used: list[str] = field(default_factory=list)
    source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly view."""
        return {
            "trajectory_id": self.trajectory_id,
            "task_id": self.task_id,
            "environment": self.environment,
            "policy_version": self.policy_version,
            "termination_reason": self.termination_reason,
            "solved": self.solved,
            "reward": self.reward,
            "tokens": self.tokens,
            "model_tokens": self.model_tokens,
            "critical_tokens": self.critical_tokens,
            "context_tokens": self.context_tokens,
            "turns": self.turns,
            "valid_tool_calls": self.valid_tool_calls,
            "invalid_tool_calls": self.invalid_tool_calls,
            "generated_tokens": self.generated_tokens,
            "tokens_by_span_type": self.tokens_by_span_type,
            "tools_used": self.tools_used,
            "source": self.source,
        }


@dataclass
class FileSummary:
    """Aggregate inspection record for a trajectory file."""

    path: str
    trajectories: int = 0
    tokens: int = 0
    model_tokens: int = 0
    critical_tokens: int = 0
    context_tokens: int = 0
    solved: int = 0
    graded: int = 0
    policy_versions: list[int] = field(default_factory=list)
    termination_reasons: dict[str, int] = field(default_factory=dict)
    tokens_by_span_type: dict[str, int] = field(default_factory=dict)
    tools_used: dict[str, int] = field(default_factory=dict)
    tokenizer_fingerprints: list[str] = field(default_factory=list)
    model_ids: list[str] = field(default_factory=list)
    samples: list[TrajectorySummary] = field(default_factory=list)

    @property
    def success_rate(self) -> float | None:
        """Fraction solved among graded trajectories."""
        return (self.solved / self.graded) if self.graded else None

    @property
    def model_token_fraction(self) -> float:
        """Share of tokens that are trainable."""
        return (self.model_tokens / self.tokens) if self.tokens else 0.0

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly view."""
        return {
            "path": self.path,
            "trajectories": self.trajectories,
            "tokens": self.tokens,
            "model_tokens": self.model_tokens,
            "critical_tokens": self.critical_tokens,
            "context_tokens": self.context_tokens,
            "model_token_fraction": self.model_token_fraction,
            "graded": self.graded,
            "solved": self.solved,
            "success_rate": self.success_rate,
            "policy_versions": self.policy_versions,
            "termination_reasons": self.termination_reasons,
            "tokens_by_span_type": self.tokens_by_span_type,
            "tools_used": self.tools_used,
            "tokenizer_fingerprints": self.tokenizer_fingerprints,
            "model_ids": self.model_ids,
            "provenance_check": {
                "trainable_span_types": sorted(s.value for s in MODEL_GENERATED_SPAN_TYPES),
                "critical_span_types": sorted(s.value for s in CRITICAL_SPAN_TYPES),
                "context_span_types_excluded_from_loss": sorted(
                    set(self.tokens_by_span_type)
                    - {s.value for s in MODEL_GENERATED_SPAN_TYPES}
                ),
            },
            "samples": [s.to_dict() for s in self.samples],
        }


def summarize_trajectory(traj: Trajectory) -> TrajectorySummary:
    """Summarize one trajectory."""
    valid = sum(1 for t in traj.turns if t.tool_call is not None and t.tool_call.valid)
    invalid = sum(1 for t in traj.turns if t.tool_call is not None and not t.tool_call.valid)
    tools = sorted(
        {t.tool_call.name for t in traj.turns if t.tool_call is not None and t.tool_call.valid}
    )
    model_tokens = sum(traj.model_generated_mask)
    return TrajectorySummary(
        trajectory_id=traj.trajectory_id,
        task_id=traj.task_id,
        environment=traj.environment,
        policy_version=traj.policy_version,
        termination_reason=traj.termination_reason.value,
        solved=(traj.verification.solved if traj.verification else None),
        reward=(traj.verification.reward if traj.verification else None),
        tokens=traj.length,
        model_tokens=model_tokens,
        critical_tokens=sum(traj.critical_mask),
        context_tokens=traj.length - model_tokens,
        turns=len(traj.turns),
        valid_tool_calls=valid,
        invalid_tool_calls=invalid,
        generated_tokens=traj.generated_token_count,
        tokens_by_span_type=traj.token_counts_by_span_type(),
        tools_used=tools,
        source=str(traj.metadata.get("source")) if traj.metadata.get("source") else None,
    )


def summarize_file(
    path: str | Path, *, limit: int = 5, trajectory_id: str | None = None
) -> FileSummary:
    """Validate a trajectory JSONL file and summarize it.

    Every record is schema-validated, so a mask that disagrees with its span
    partition makes this command fail rather than print a reassuring summary.
    """
    summary = FileSummary(path=str(path))
    versions: set[int] = set()
    fingerprints: set[str] = set()
    models: set[str] = set()
    for traj in iter_trajectories(path):
        if trajectory_id is not None and traj.trajectory_id != trajectory_id:
            continue
        record = summarize_trajectory(traj)
        summary.trajectories += 1
        summary.tokens += record.tokens
        summary.model_tokens += record.model_tokens
        summary.critical_tokens += record.critical_tokens
        summary.context_tokens += record.context_tokens
        if record.solved is not None:
            summary.graded += 1
            summary.solved += int(record.solved)
        versions.add(record.policy_version)
        fingerprints.add(traj.tokenizer_fingerprint)
        models.add(traj.model_id)
        summary.termination_reasons[record.termination_reason] = (
            summary.termination_reasons.get(record.termination_reason, 0) + 1
        )
        for name, count in record.tokens_by_span_type.items():
            summary.tokens_by_span_type[name] = summary.tokens_by_span_type.get(name, 0) + count
        for tool in record.tools_used:
            summary.tools_used[tool] = summary.tools_used.get(tool, 0) + 1
        if len(summary.samples) < limit:
            summary.samples.append(record)
    summary.policy_versions = sorted(versions)
    summary.tokenizer_fingerprints = sorted(fingerprints)
    summary.model_ids = sorted(models)
    return summary


def iter_spans_for_display(
    path: str | Path, trajectory_id: str
) -> Iterator[dict[str, Any]]:
    """Yield display rows for one trajectory's spans."""
    for traj in iter_trajectories(path):
        if traj.trajectory_id != trajectory_id:
            continue
        for span in traj.spans:
            yield {
                "span_type": span.span_type.value,
                "start": span.start,
                "end": span.end,
                "tokens": span.length,
                "turn_id": span.turn_id,
                "in_loss": span.is_model_generated,
                "critical": span.is_critical,
                "tool_name": span.tool_name,
                "text": span.text,
            }
        return
