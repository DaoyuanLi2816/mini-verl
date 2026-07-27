"""Benchmark configuration and result schemas.

The result schema doubles as the contribution format: ``miniverl benchmark``
writes it, ``miniverl export-benchmark`` sanitizes it for a pull request, and
``benchmarks/schema/benchmark-result.schema.json`` is generated from the same
Pydantic model so the two can never drift.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from miniverl.errors import ConfigError

__all__ = [
    "BENCHMARK_SCHEMA_VERSION",
    "BenchmarkArm",
    "BenchmarkConfig",
    "ArmResult",
    "BenchmarkResult",
    "json_schema",
]

BENCHMARK_SCHEMA_VERSION = 1


class BenchmarkArm(BaseModel):
    """One arm of a matched comparison."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=64)
    description: str = ""
    #: Deep-merged into the base recipe. Only keys that the arm is *supposed* to
    #: differ in should appear here -- everything else stays matched.
    overrides: dict[str, Any] = Field(default_factory=dict)


class BenchmarkConfig(BaseModel):
    """A matched-budget benchmark specification."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = BENCHMARK_SCHEMA_VERSION
    name: str = Field(min_length=1, max_length=80)
    description: str = ""
    #: Path to a recipe YAML (relative to the benchmark file) or an inline mapping.
    base: str | dict[str, Any]
    #: Shared SFT cold start. Every arm resumes from the *same* resulting weights,
    #: which is what makes the comparison a comparison.
    cold_start_cycles: int = Field(default=0, ge=0, le=100000)
    eval_split: str = Field(default="test", pattern="^(train|eval|test)$")
    seeds: list[int] = Field(default_factory=lambda: [1234], min_length=1)
    arms: list[BenchmarkArm] = Field(min_length=1)
    output_dir: str = "runs/benchmarks"

    @model_validator(mode="after")
    def _validate(self) -> BenchmarkConfig:
        if self.schema_version != BENCHMARK_SCHEMA_VERSION:
            raise ValueError(
                f"benchmark schema_version {self.schema_version} is not supported "
                f"(expected {BENCHMARK_SCHEMA_VERSION})"
            )
        names = [a.name for a in self.arms]
        if len(set(names)) != len(names):
            raise ValueError("benchmark arm names must be unique")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> BenchmarkConfig:
        """Load a benchmark specification."""
        p = Path(path)
        if not p.is_file():
            raise ConfigError(
                f"benchmark config not found: {p}",
                hint="see benchmarks/configs/ for examples",
            )
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ConfigError(f"{p} must contain a YAML mapping at the top level")
        config = cls.model_validate(raw)
        if isinstance(config.base, str):
            base_path = (p.parent / config.base).resolve()
            config = config.model_copy(update={"base": str(base_path)})
        return config


class ArmResult(BaseModel):
    """Measured outcome for one arm at one seed."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    mode: str
    seed: int
    run_id: str
    run_dir: str
    loss_mode: str
    divergence: str
    selector: str
    top_k: int
    optimizer_steps: int
    policy_version: int
    tasks: int
    success_rate: float
    avg_turns: float
    avg_tool_calls: float
    invalid_tool_call_rate: float
    generated_tokens_per_task: float
    tokens_per_solved_task: float | None = None
    selected_training_tokens: int
    teacher_queried_position_ratio: float | None = None
    cache_bytes: int | None = None
    cache_compression_ratio: float | None = None
    peak_allocated_bytes: int | None = None
    peak_reserved_bytes: int | None = None
    seconds: float
    baseline_success_rate: float | None = None
    measurement_status: str = "measured"


class BenchmarkResult(BaseModel):
    """Full benchmark output, including everything that was held constant."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = BENCHMARK_SCHEMA_VERSION
    miniverl_version: str
    name: str
    description: str = ""
    created_at: str
    git_commit: str | None = None
    hardware: dict[str, Any] = Field(default_factory=dict)
    software: dict[str, Any] = Field(default_factory=dict)
    controlled: dict[str, Any] = Field(default_factory=dict)
    arms: list[ArmResult] = Field(default_factory=list)
    notes: str = ""
    seeds: list[int] = Field(default_factory=list)

    def by_arm(self) -> dict[str, list[ArmResult]]:
        """Group results by arm name."""
        grouped: dict[str, list[ArmResult]] = {}
        for arm in self.arms:
            grouped.setdefault(arm.name, []).append(arm)
        return grouped

    def aggregate(self) -> list[dict[str, Any]]:
        """Mean success rate per arm, with the seed count made explicit."""
        rows = []
        for name, results in self.by_arm().items():
            rates = [r.success_rate for r in results]
            rows.append(
                {
                    "name": name,
                    "mode": results[0].mode,
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
    """JSON Schema for :class:`BenchmarkResult`, for the contribution template."""
    schema = BenchmarkResult.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["title"] = "miniVERL benchmark result"
    schema["$id"] = (
        "https://raw.githubusercontent.com/DaoyuanLi2816/mini-verl/main/"
        "benchmarks/schema/benchmark-result.schema.json"
    )
    return schema
