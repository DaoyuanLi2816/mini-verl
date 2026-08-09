"""One typed per-example record shared by every external endpoint.

Every endpoint produces the same row shape, so results can be validated,
compared and aggregated without per-benchmark special cases. Two rules are
structural rather than conventional:

* a missing or inapplicable metric is ``None`` with a stated reason. It is
  never coerced to zero, because zero is a measurement and ``None`` is the
  absence of one;
* generated text is never stored. Only its digest is, so a task-level artifact
  can prove which text was scored without republishing a jailbreak completion.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

__all__ = [
    "RECORD_SCHEMA_VERSION",
    "EvaluationStatus",
    "TaskRecord",
    "digest_text",
]

RECORD_SCHEMA_VERSION = 1

EvaluationStatus = Literal["evaluated", "not_applicable", "failed", "skipped"]


def digest_text(text: str) -> str:
    """Stable digest of generated text, so it can be proven without publishing."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class TaskRecord:
    """One (checkpoint, endpoint, task) evaluation."""

    # --- identity of what was evaluated
    endpoint_id: str
    category: str
    dataset: str
    dataset_revision: str
    split: str
    task_id: str
    subset: str | None

    # --- identity of the thing being measured
    checkpoint_id: str
    checkpoint_digest: str
    method: str
    seed: int | None

    # --- what it produced
    generation_config_digest: str
    output_digest: str | None
    output_tokens: int | None

    # --- what the evaluator said
    score: float | None
    subscores: dict[str, Any] = field(default_factory=dict)
    evaluator_id: str = ""
    evaluator_revision: str | None = None
    evaluator_config_digest: str | None = None

    # --- why it says what it says
    status: EvaluationStatus = "evaluated"
    not_applicable_reason: str | None = None

    # --- cost
    generation_seconds: float | None = None
    evaluation_seconds: float | None = None
    prompt_tokens: int | None = None

    # --- publication scope
    publication: str = "aggregate_and_digest_only"
    schema_version: int = RECORD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.status == "evaluated" and self.score is None:
            raise ValueError(
                f"{self.endpoint_id}/{self.task_id}: status 'evaluated' needs a score; "
                "use status 'not_applicable' with a reason instead of a null score"
            )
        if self.status != "evaluated" and self.score is not None:
            raise ValueError(
                f"{self.endpoint_id}/{self.task_id}: status {self.status!r} must not carry "
                "a score; a score that was not measured is not a score"
            )
        if self.status == "not_applicable" and not self.not_applicable_reason:
            raise ValueError(
                f"{self.endpoint_id}/{self.task_id}: not_applicable needs a stated reason"
            )

    def to_json_row(self) -> dict[str, Any]:
        """Serialise for the task-level JSONL artifact."""
        return asdict(self)

    @classmethod
    def not_applicable(
        cls,
        *,
        reason: str,
        **fields: Any,
    ) -> TaskRecord:
        """Build an explicitly unmeasured record. Never a zero-scored one."""
        fields.setdefault("subset", None)
        fields.setdefault("seed", None)
        fields.setdefault("output_digest", None)
        fields.setdefault("output_tokens", None)
        return cls(
            score=None,
            status="not_applicable",
            not_applicable_reason=reason,
            **fields,
        )


def validate_rows(rows: list[dict[str, Any]]) -> list[str]:
    """Structural problems in a task-level artifact, empty when it is sound."""
    problems: list[str] = []
    seen: set[tuple[str, str, str, int | None]] = set()
    for index, row in enumerate(rows):
        if row.get("schema_version") != RECORD_SCHEMA_VERSION:
            problems.append(f"row {index}: schema_version is not {RECORD_SCHEMA_VERSION}")
        status = row.get("status")
        if status == "evaluated" and row.get("score") is None:
            problems.append(f"row {index}: evaluated without a score")
        if status != "evaluated" and row.get("score") is not None:
            problems.append(f"row {index}: {status} carries a score")
        if status == "not_applicable" and not row.get("not_applicable_reason"):
            problems.append(f"row {index}: not_applicable without a reason")
        key = (
            str(row.get("endpoint_id")),
            str(row.get("task_id")),
            str(row.get("checkpoint_id")),
            row.get("seed"),
        )
        if key in seen:
            problems.append(f"row {index}: duplicate {key}")
        seen.add(key)
    return problems


def config_digest(config: dict[str, Any]) -> str:
    """Digest of a generation or evaluator configuration."""
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
