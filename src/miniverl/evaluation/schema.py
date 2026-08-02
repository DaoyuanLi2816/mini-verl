"""Versioned benchmark configuration and result schemas.

Version 2 records resolved configuration provenance and cumulative accounting.
Version 3 adds RecoveryBench's preregistration, recovery, budget and task-level
artifact provenance. Public version-1 and version-2 artifacts remain readable
and are never reinterpreted or rewritten in place.
"""

from __future__ import annotations

import math
from pathlib import Path, PurePath
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from miniverl.errors import ConfigError

__all__ = [
    "BENCHMARK_SCHEMA_VERSION",
    "RECOVERY_BENCHMARK_SCHEMA_VERSION",
    "LEGACY_BENCHMARK_SCHEMA_VERSION",
    "BenchmarkArm",
    "BenchmarkConfig",
    "ArmResult",
    "BenchmarkResult",
    "json_schema",
    "recovery_json_schema",
    "finite_or_none",
]

LEGACY_BENCHMARK_SCHEMA_VERSION = 1
BENCHMARK_SCHEMA_VERSION = 2
RECOVERY_BENCHMARK_SCHEMA_VERSION = 3


def _resolve_local_adapter_path(overrides: dict[str, Any], parent: Path) -> None:
    """Resolve a benchmark-local teacher adapter without touching Hub IDs."""
    models = overrides.get("models")
    if not isinstance(models, dict):
        return
    teacher = models.get("teacher")
    if not isinstance(teacher, dict):
        return
    adapter = teacher.get("adapter")
    if not isinstance(adapter, dict) or adapter.get("source", "local") != "local":
        return
    raw_path = adapter.get("path")
    if isinstance(raw_path, str) and raw_path and not Path(raw_path).is_absolute():
        adapter["path"] = str((parent / raw_path).resolve())


def finite_or_none(value: object) -> float | None:
    """Coerce to ``float``, mapping undefined and non-finite values to ``None``."""
    if value is None:
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


class BenchmarkArm(BaseModel):
    """One arm of a controlled comparison."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=64)
    description: str = ""
    overrides: dict[str, Any] = Field(default_factory=dict)


class BenchmarkConfig(BaseModel):
    """Versioned benchmark specification with explicit cold-start semantics."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[2, 3] = 2
    name: str = Field(min_length=1, max_length=80)
    description: str = ""
    base: str | dict[str, Any]
    common_overrides: dict[str, Any] = Field(default_factory=dict)
    cold_start_overrides: dict[str, Any] = Field(default_factory=dict)
    cold_start_cycles: int = Field(default=0, ge=0, le=100000)
    cold_start_checkpoint_template: str | None = None
    frozen_dataset_template: str | None = None
    allowed_differences: list[str] = Field(default_factory=list)
    budget_axis: Literal["optimizer_steps", "selected_training_tokens", "wall_time"] = (
        "optimizer_steps"
    )
    eval_split: str = Field(default="test", pattern="^(train|eval|test)$")
    seeds: list[int] = Field(default_factory=lambda: [1234], min_length=1)
    arms: list[BenchmarkArm] = Field(min_length=1)
    output_dir: str = "runs/benchmarks"

    # Schema-v3 RecoveryBench provenance. These stay optional for schema-v2
    # calculator specifications and are mandatory as a complete set for v3.
    preregistration_sha: str | None = None
    preregistration_digest: str | None = None
    hypothesis_ids: list[str] = Field(default_factory=list)
    task_schedule_digest: str | None = None
    template_registry_version: int | None = None
    template_registry_digest: str | None = None
    selected_teacher_candidate: dict[str, Any] | None = None
    teacher_gate_results: list[dict[str, Any]] = Field(default_factory=list)
    teacher_preparation_cost: dict[str, Any] | None = None
    frozen_dataset_identity: dict[str, Any] = Field(default_factory=dict)
    budget_view: str | None = None
    stop_criterion: dict[str, Any] | None = None
    task_level_artifacts: list[dict[str, Any]] = Field(default_factory=list)
    result_analysis_version: str | None = None
    invalidation_status: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _validate(self) -> BenchmarkConfig:
        names = [a.name for a in self.arms]
        if len(set(names)) != len(names):
            raise ValueError("benchmark arm names must be unique")
        if any(
            not path or path.startswith(".") or path.endswith(".")
            for path in self.allowed_differences
        ):
            raise ValueError("allowed_differences entries must be non-empty dotted config paths")
        for field_name in ("cold_start_checkpoint_template", "frozen_dataset_template"):
            template = getattr(self, field_name)
            if template is None:
                continue
            try:
                template.format(seed=self.seeds[0])
            except (KeyError, ValueError) as exc:
                raise ValueError(f"{field_name} may contain only the {{seed}} placeholder") from exc
            if len(self.seeds) > 1 and "{seed}" not in template:
                raise ValueError(f"{field_name} must contain {{seed}} for a multi-seed benchmark")
        if self.schema_version == RECOVERY_BENCHMARK_SCHEMA_VERSION:
            required = {
                "preregistration_sha": self.preregistration_sha,
                "preregistration_digest": self.preregistration_digest,
                "hypothesis_ids": self.hypothesis_ids,
                "task_schedule_digest": self.task_schedule_digest,
                "template_registry_version": self.template_registry_version,
                "template_registry_digest": self.template_registry_digest,
                "selected_teacher_candidate": self.selected_teacher_candidate,
                "teacher_gate_results": self.teacher_gate_results,
                "teacher_preparation_cost": self.teacher_preparation_cost,
                "budget_view": self.budget_view,
                "stop_criterion": self.stop_criterion,
                "result_analysis_version": self.result_analysis_version,
                "invalidation_status": self.invalidation_status,
            }
            missing = [name for name, value in required.items() if value is None or value == []]
            if missing:
                raise ValueError("benchmark config schema v3 requires: " + ", ".join(missing))
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> BenchmarkConfig:
        """Load a v2 benchmark specification and resolve its base recipe path."""
        p = Path(path)
        if not p.is_file():
            raise ConfigError(
                f"benchmark config not found: {p}",
                hint="see benchmarks/configs/ for examples",
            )
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ConfigError(f"{p} must contain a YAML mapping at the top level")
        for key in ("cold_start_checkpoint_template", "frozen_dataset_template"):
            template = raw.get(key)
            if isinstance(template, str) and template and not Path(template).is_absolute():
                raw[key] = str((p.parent / template).resolve())
        for key in ("common_overrides", "cold_start_overrides"):
            overrides = raw.get(key)
            if isinstance(overrides, dict):
                _resolve_local_adapter_path(overrides, p.parent)
        arms = raw.get("arms")
        if isinstance(arms, list):
            for arm in arms:
                if isinstance(arm, dict) and isinstance(arm.get("overrides"), dict):
                    _resolve_local_adapter_path(arm["overrides"], p.parent)
        config = cls.model_validate(raw)
        if isinstance(config.base, str):
            base_path = (p.parent / config.base).resolve()
            config = config.model_copy(update={"base": str(base_path)})
        return config


class ArmResult(BaseModel):
    """Measured outcome for one arm at one seed.

    The compatibility normalizer maps the two renamed v1 fields into their v2
    cumulative names. Properties retain the old Python accessor names without
    serializing duplicate fields into new artifacts.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    mode: str
    seed: int
    run_id: str
    run_dir: str
    objective: str = "legacy_unreported"
    opd_freshness: str | None = None
    loss_mode: str | None = None
    divergence: str | None = None
    selector: str | None = None
    top_k: int | None = None

    resolved_config_digest: str | None = None
    structured_diff: list[dict[str, Any]] = Field(default_factory=list)
    student_model_id: str | None = None
    student_model_revision: str | None = None
    teacher_model_id: str | None = None
    teacher_model_revision: str | None = None
    teacher_adapter: dict[str, Any] | None = None
    tokenizer_fingerprint: str | None = None
    teacher_context_mode: str | None = None

    # ``structured_diff`` and ``resolved_config_digest`` are retained with
    # their schema-v2 meaning for existing readers: the fully defaulted,
    # pre-allocation arm config compared with the common config. The additive
    # fields below separate scientific treatments from harness bookkeeping and
    # runtime decisions without requiring a schema-v3 migration.
    declared_config_digest: str | None = None
    runtime_resolved_config_digest: str | None = None
    scientific_config_diff: list[dict[str, Any]] = Field(default_factory=list)
    runtime_resolution_diff: list[dict[str, Any]] = Field(default_factory=list)
    harness_config_diff: list[dict[str, Any]] = Field(default_factory=list)

    optimizer_steps: int
    policy_version: int
    total_trajectories: int = 0
    generated_training_tokens_total: int = 0
    selected_training_tokens_total: int = 0
    model_generated_training_tokens_total: int = 0
    selected_position_ratio: float | None = None
    teacher_queried_positions_total: int | None = None
    teacher_queried_position_ratio: float | None = None

    tasks: int
    success_rate: float
    strict_task_success_rate: float | None = None
    lenient_diagnostic_success_rate: float | None = None
    avg_turns: float
    avg_tool_calls: float
    assistant_turns: int | None = None
    emitted_tool_calls: int | None = None
    parsed_tool_calls: int | None = None
    tool_execution_successes: int | None = None
    tool_execution_errors: int | None = None
    unknown_tool_calls: int | None = None
    parse_errors: int | None = None
    repeated_call_terminations: int | None = None
    final_answers_emitted: int | None = None
    final_answers_format_valid: int | None = None
    final_answers_verified: int | None = None
    parse_valid_tool_call_rate: float | None = None
    tool_execution_success_rate: float | None = None
    tool_execution_error_rate: float | None = None
    # Deprecated aliases for schema-v1/v2 benchmark readers.
    tool_call_count: int | None = None
    valid_tool_call_rate: float | None = None
    invalid_tool_call_rate: float
    final_answer_format_validity_rate: float | None = None
    protocol_token_accuracy: float | None = None
    generated_tokens_per_task: float
    tokens_per_solved_task: float | None = None

    cache_current_bytes: int | None = None
    cache_bytes_written_total: int | None = None
    cache_compression_ratio: float | None = None
    peak_allocated_bytes: int | None = None
    peak_reserved_bytes: int | None = None
    train_seconds: float | None = None
    evaluation_seconds: float | None = None
    wall_seconds: float
    baseline_success_rate: float | None = None
    measurement_status: dict[str, str] | str = Field(default_factory=dict)
    recovery_metrics: dict[str, Any] = Field(default_factory=dict)
    frozen_dataset_identity: dict[str, Any] | None = None
    stop_criterion: dict[str, Any] | None = None
    overshoot: dict[str, Any] = Field(default_factory=dict)
    task_level_artifact: dict[str, Any] | None = None

    @model_validator(mode="before")
    @classmethod
    def _read_v1_names(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        if "selected_training_tokens_total" not in payload:
            payload["selected_training_tokens_total"] = int(
                payload.pop("selected_training_tokens", 0)
            )
        else:
            payload.pop("selected_training_tokens", None)
        if "wall_seconds" not in payload:
            payload["wall_seconds"] = float(payload.pop("seconds", 0.0))
        else:
            payload.pop("seconds", None)
        if "cache_current_bytes" not in payload:
            payload["cache_current_bytes"] = payload.pop("cache_bytes", None)
        else:
            payload.pop("cache_bytes", None)
        return payload

    @field_validator("run_dir")
    @classmethod
    def _strip_to_directory_name(cls, value: str) -> str:
        return PurePath(value.replace("\\", "/")).name or value

    @property
    def selected_training_tokens(self) -> int:
        """v1 accessor retained for downstream readers."""
        return self.selected_training_tokens_total

    @property
    def seconds(self) -> float:
        """v1 accessor retained for downstream readers."""
        return self.wall_seconds

    @property
    def cache_bytes(self) -> int | None:
        """v1 accessor retained for downstream readers."""
        return self.cache_current_bytes


class BenchmarkResult(BaseModel):
    """Full benchmark output with v1/v2 read compatibility and v3 provenance."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1, 2, 3] = 2
    miniverl_version: str
    name: str
    description: str = ""
    created_at: str
    git_commit: str | None = None
    invocation: list[str] | None = None
    budget_axis: str | None = None
    hardware: dict[str, Any] = Field(default_factory=dict)
    software: dict[str, Any] = Field(default_factory=dict)
    cold_start: dict[str, Any] | None = None
    common_resolved_config: dict[str, Any] | None = None
    common_resolved_config_digest: str | None = None
    common_declared_config: dict[str, Any] | None = None
    common_declared_config_digest: str | None = None
    controlled: dict[str, Any] = Field(default_factory=dict)
    arms: list[ArmResult] = Field(default_factory=list)
    notes: str = ""
    seeds: list[int] = Field(default_factory=list)

    preregistration_sha: str | None = None
    preregistration_digest: str | None = None
    hypothesis_ids: list[str] = Field(default_factory=list)
    task_schedule_digest: str | None = None
    template_registry_version: int | None = None
    template_registry_digest: str | None = None
    selected_teacher_candidate: dict[str, Any] | None = None
    teacher_gate_results: list[dict[str, Any]] = Field(default_factory=list)
    teacher_preparation_cost: dict[str, Any] | None = None
    frozen_dataset_identity: dict[str, Any] = Field(default_factory=dict)
    budget_view: str | None = None
    stop_criterion: dict[str, Any] | None = None
    overshoot: dict[str, Any] = Field(default_factory=dict)
    recovery_metrics: dict[str, Any] = Field(default_factory=dict)
    task_level_artifacts: list[dict[str, Any]] = Field(default_factory=list)
    result_analysis_version: str | None = None
    invalidation_status: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _require_v2_provenance(self) -> BenchmarkResult:
        if self.schema_version >= BENCHMARK_SCHEMA_VERSION:
            missing = [
                name
                for name, value in (
                    ("invocation", self.invocation),
                    ("budget_axis", self.budget_axis),
                    ("cold_start", self.cold_start),
                    ("common_resolved_config", self.common_resolved_config),
                    ("common_resolved_config_digest", self.common_resolved_config_digest),
                )
                if value is None
            ]
            if missing:
                raise ValueError(
                    "benchmark schema v2 requires provenance fields: " + ", ".join(missing)
                )
        if self.schema_version == RECOVERY_BENCHMARK_SCHEMA_VERSION:
            required = {
                "preregistration_sha": self.preregistration_sha,
                "preregistration_digest": self.preregistration_digest,
                "hypothesis_ids": self.hypothesis_ids,
                "task_schedule_digest": self.task_schedule_digest,
                "template_registry_version": self.template_registry_version,
                "template_registry_digest": self.template_registry_digest,
                "selected_teacher_candidate": self.selected_teacher_candidate,
                "teacher_gate_results": self.teacher_gate_results,
                "teacher_preparation_cost": self.teacher_preparation_cost,
                "budget_view": self.budget_view,
                "stop_criterion": self.stop_criterion,
                "recovery_metrics": self.recovery_metrics,
                "task_level_artifacts": self.task_level_artifacts,
                "result_analysis_version": self.result_analysis_version,
                "invalidation_status": self.invalidation_status,
            }
            missing = [name for name, value in required.items() if value is None or value == []]
            if missing:
                raise ValueError(
                    "benchmark schema v3 requires provenance fields: " + ", ".join(missing)
                )
        return self

    def by_arm(self) -> dict[str, list[ArmResult]]:
        grouped: dict[str, list[ArmResult]] = {}
        for arm in self.arms:
            grouped.setdefault(arm.name, []).append(arm)
        return grouped

    def aggregate(self) -> list[dict[str, Any]]:
        """Mean success per arm, explicitly retaining the seed count and range."""
        rows = []
        for name, results in self.by_arm().items():
            rates = [r.success_rate for r in results]
            rows.append(
                {
                    "name": name,
                    "mode": results[0].mode,
                    "objective": results[0].objective,
                    "seeds": len(results),
                    "success_rate_mean": sum(rates) / len(rates),
                    "success_rate_min": min(rates),
                    "success_rate_max": max(rates),
                    "optimizer_steps": results[0].optimizer_steps,
                    "single_seed": len(results) == 1,
                }
            )
        return rows


def json_schema() -> dict[str, Any]:
    """JSON Schema accepting preserved v1/v2 and RecoveryBench-v3 results."""
    schema = BenchmarkResult.model_json_schema()
    # Pydantic's pre-validator migrates the three renamed v1 fields at runtime,
    # but JSON Schema cannot see that Python hook. Describe the legacy spellings
    # explicitly so the public validator accepts both preserved v1 artifacts
    # and new v2 output without weakening the runtime model.
    arm = schema["$defs"]["ArmResult"]
    arm["properties"].update(
        {
            "selected_training_tokens": {
                "minimum": 0,
                "title": "Selected Training Tokens",
                "type": "integer",
            },
            "seconds": {
                "minimum": 0,
                "title": "Seconds",
                "type": "number",
            },
            "cache_bytes": {
                "anyOf": [
                    {"minimum": 0, "type": "integer"},
                    {"type": "null"},
                ],
                "default": None,
                "title": "Cache Bytes",
            },
        }
    )
    arm["required"] = [field for field in arm.get("required", []) if field != "wall_seconds"]
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["title"] = "miniVERL benchmark result"
    schema["$id"] = (
        "https://raw.githubusercontent.com/DaoyuanLi2816/mini-verl/main/"
        "benchmarks/schema/benchmark-result.schema.json"
    )
    return schema


def recovery_json_schema() -> dict[str, Any]:
    """RecoveryBench-v3 publication view with legacy read compatibility."""
    schema = json_schema()
    schema["title"] = "miniVERL RecoveryBench result"
    schema["$id"] = (
        "https://raw.githubusercontent.com/DaoyuanLi2816/mini-verl/main/"
        "benchmarks/schema/recoverybench-result.schema.json"
    )
    return schema
