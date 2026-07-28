"""Versioned benchmark configuration and result schemas.

Version 2 records resolved configuration provenance and cumulative accounting.
The public version-1 artifacts remain readable, but new benchmark executions
always write version 2 and never rewrite old measurements in place.
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
    "LEGACY_BENCHMARK_SCHEMA_VERSION",
    "BenchmarkArm",
    "BenchmarkConfig",
    "ArmResult",
    "BenchmarkResult",
    "json_schema",
    "finite_or_none",
]

LEGACY_BENCHMARK_SCHEMA_VERSION = 1
BENCHMARK_SCHEMA_VERSION = 2


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
    """Benchmark-v2 specification with explicit cold-start semantics."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[2] = 2
    name: str = Field(min_length=1, max_length=80)
    description: str = ""
    base: str | dict[str, Any]
    common_overrides: dict[str, Any] = Field(default_factory=dict)
    cold_start_overrides: dict[str, Any] = Field(default_factory=dict)
    cold_start_cycles: int = Field(default=0, ge=0, le=100000)
    allowed_differences: list[str] = Field(default_factory=list)
    budget_axis: Literal["optimizer_steps", "selected_training_tokens", "wall_time"] = (
        "optimizer_steps"
    )
    eval_split: str = Field(default="test", pattern="^(train|eval|test)$")
    seeds: list[int] = Field(default_factory=lambda: [1234], min_length=1)
    arms: list[BenchmarkArm] = Field(min_length=1)
    output_dir: str = "runs/benchmarks"

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
    """Full benchmark output, with v1 read compatibility and v2 provenance."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1, 2] = 2
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

    @model_validator(mode="after")
    def _require_v2_provenance(self) -> BenchmarkResult:
        if self.schema_version == BENCHMARK_SCHEMA_VERSION:
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
    """JSON Schema accepting preserved v1 and newly generated v2 results."""
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
