"""Regression tests for benchmark-v2 provenance and cumulative accounting."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from miniverl.config import TrainingMode
from miniverl.errors import ConfigError
from miniverl.evaluation.benchmark import (
    _training_accounting,
    portable_payload,
    resolve_benchmark_configs,
    run_benchmark,
)
from miniverl.evaluation.schema import BenchmarkConfig, BenchmarkResult
from miniverl.utils.runs import JsonlWriter


def _base() -> dict[str, Any]:
    return {
        "models": {
            "student": {"model_id": "toy-student"},
            "teacher": {"model_id": "toy-teacher"},
        },
        "environment": {
            "name": "calculator",
            "difficulty": "medium",
            "test_tasks": 48,
        },
        "train": {
            "rollouts_per_cycle": 8,
            "gradient_accumulation_steps": 8,
            "learning_rate": 1.0e-4,
            "lr_schedule": "cosine",
        },
    }


def _spec(**updates: Any) -> BenchmarkConfig:
    payload: dict[str, Any] = {
        "schema_version": 2,
        "name": "provenance-regression",
        "base": _base(),
        "common_overrides": {
            "environment": {"difficulty": "hard", "test_tasks": 24},
            "train": {"learning_rate": 5.0e-5, "lr_schedule": "constant"},
        },
        "cold_start_overrides": {"environment": {"difficulty": "hard"}},
        "cold_start_cycles": 3,
        "allowed_differences": ["run.mode", "train.cycles"],
        "budget_axis": "optimizer_steps",
        "arms": [
            {
                "name": "cold-start-only",
                "overrides": {"run": {"mode": "sft"}, "train": {"cycles": 0}},
            },
            {
                "name": "opd",
                "overrides": {"run": {"mode": "opd"}, "train": {"cycles": 3}},
            },
        ],
    }
    payload.update(updates)
    return BenchmarkConfig.model_validate(payload)


def test_cold_start_and_arms_resolve_from_explicit_overrides() -> None:
    common, cold, arms = resolve_benchmark_configs(_spec())

    assert common.environment.difficulty == "hard"
    assert common.environment.test_tasks == 24
    assert common.train.learning_rate == pytest.approx(5.0e-5)
    assert common.train.lr_schedule.value == "constant"
    assert cold.environment.difficulty == "hard"
    assert cold.train.cycles == 3
    assert cold.run.mode is TrainingMode.SFT
    assert {arm.name: cfg.environment.difficulty for arm, cfg, _ in arms} == {
        "cold-start-only": "hard",
        "opd": "hard",
    }


def test_the_legacy_medium_to_hard_bug_cannot_hide_in_controlled_metadata() -> None:
    spec = _spec(cold_start_overrides={"environment": {"difficulty": "medium"}})
    common, cold, arms = resolve_benchmark_configs(spec)

    assert common.environment.difficulty == "hard"
    assert cold.environment.difficulty == "medium"
    assert all(cfg.environment.difficulty == "hard" for _, cfg, _ in arms)


def test_undeclared_difference_fails_during_preflight() -> None:
    spec = _spec(
        arms=[
            {
                "name": "bad",
                "overrides": {
                    "run": {"mode": "opd"},
                    "train": {"cycles": 3},
                    "environment": {"difficulty": "easy"},
                },
            }
        ]
    )
    with pytest.raises(ConfigError, match="undeclared arm differences") as excinfo:
        resolve_benchmark_configs(spec)
    assert "environment.difficulty" in str(excinfo.value)
    assert "before any model is loaded" in str(excinfo.value)


def test_local_teacher_adapter_path_is_relative_to_benchmark_file(
    tmp_path: Path,
) -> None:
    benchmark_dir = tmp_path / "benchmarks" / "configs"
    benchmark_dir.mkdir(parents=True)
    path = benchmark_dir / "adapter.yaml"
    path.write_text(
        """
schema_version: 2
name: adapter-relative
base: {}
allowed_differences: [models.teacher.adapter]
arms:
  - name: protocol-teacher
    overrides:
      models:
        teacher:
          adapter:
            path: ../../artifacts/protocol-teacher
""",
        encoding="utf-8",
    )

    config = BenchmarkConfig.from_yaml(path)
    adapter = config.arms[0].overrides["models"]["teacher"]["adapter"]
    assert Path(adapter["path"]) == (tmp_path / "artifacts" / "protocol-teacher").resolve()


def test_published_provenance_replaces_machine_local_absolute_paths() -> None:
    payload = {
        "run": {"output_dir": r"C:\Users\alice\runs"},
        "models": {
            "student": {"model_id": "Qwen/Qwen3-0.6B"},
            "teacher": {
                "adapter": {
                    "source": "local",
                    "path": r"C:\Users\alice\artifacts\protocol-teacher",
                }
            },
        },
    }
    portable = portable_payload(payload)
    assert portable["run"]["output_dir"] == "<local>/runs"
    assert portable["models"]["student"]["model_id"] == "Qwen/Qwen3-0.6B"
    assert portable["models"]["teacher"]["adapter"]["path"] == "<local>/protocol-teacher"
    assert "alice" not in json.dumps(portable).lower()


@pytest.mark.torch
def test_schema_v2_benchmark_runs_end_to_end_and_writes_provenance(tmp_path: Path) -> None:
    base = _base()
    base["environment"].update(
        {"difficulty": "easy", "train_tasks": 1, "eval_tasks": 1, "test_tasks": 1}
    )
    base["eval"] = {"enabled": False, "tasks": 1, "split": "test"}
    spec = BenchmarkConfig.model_validate(
        {
            "schema_version": 2,
            "name": "tiny-v2",
            "base": base,
            "cold_start_cycles": 0,
            "allowed_differences": ["run.mode", "train.cycles"],
            "budget_axis": "optimizer_steps",
            "seeds": [7],
            "arms": [
                {
                    "name": "cold-start-only",
                    "overrides": {"run": {"mode": "sft"}, "train": {"cycles": 0}},
                }
            ],
        }
    )

    result = run_benchmark(
        spec,
        output_dir=tmp_path / "benchmarks",
        invocation=["miniverl", "benchmark", "tiny.yaml"],
    )

    assert result.schema_version == 2
    assert result.invocation == ["miniverl", "benchmark", "tiny.yaml"]
    assert result.common_resolved_config_digest
    assert result.controlled["digest"] == result.common_resolved_config_digest
    assert result.cold_start["checkpoints"][0]["checkpoint_digest"] is None
    assert len(result.arms) == 1
    arm = result.arms[0]
    assert arm.objective == "sft_cross_entropy"
    assert arm.teacher_model_id is None
    assert arm.teacher_queried_positions_total is None
    assert arm.optimizer_steps == 0
    assert arm.wall_seconds >= arm.evaluation_seconds
    persisted_text = (tmp_path / "benchmarks" / "tiny-v2.json").read_text(encoding="utf-8")
    assert str(tmp_path).lower() not in persisted_text.lower()
    persisted = BenchmarkResult.model_validate_json(persisted_text)
    assert persisted.common_resolved_config_digest == result.common_resolved_config_digest


def _metrics(tmp_path: Path, rows: list[dict[str, Any]]) -> Any:
    path = tmp_path / "metrics.jsonl"
    writer = JsonlWriter(path)
    for row in rows:
        writer.write(row)
    return SimpleNamespace(paths=SimpleNamespace(metrics=path))


def test_accounting_sums_numerators_and_denominators_across_cycles(tmp_path: Path) -> None:
    trainer = _metrics(
        tmp_path,
        [
            {
                "phase": "opd_cycle",
                "rollouts": {"rollouts": 2, "generated_tokens": 10},
                "selection": {"selected_model_tokens": 2, "total_model_tokens": 4},
            },
            {
                "phase": "opd_cycle",
                "rollouts": {"rollouts": 5, "generated_tokens": 90},
                "selection": {"selected_model_tokens": 9, "total_model_tokens": 10},
            },
        ],
    )
    result = _training_accounting(trainer, TrainingMode.OPD)
    assert result["total_trajectories"] == 7
    assert result["generated_training_tokens_total"] == 100
    assert result["model_generated_training_tokens_total"] == 100
    assert result["selected_training_tokens_total"] == 11
    assert result["selected_position_ratio"] == pytest.approx(11 / 14)
    assert result["teacher_queried_positions_total"] == 11
    assert result["teacher_queried_position_ratio"] == pytest.approx(11 / 14)


def test_sft_and_zero_step_accounting_are_mode_correct(tmp_path: Path) -> None:
    sft = _metrics(
        tmp_path / "sft",
        [
            {
                "phase": "sft_cycle",
                "rollouts": {"rollouts": 3, "generated_tokens": 30},
                "selection": {"selected_model_tokens": 12, "total_model_tokens": 20},
                "cache": {"actual_bytes": 999},
            }
        ],
    )
    result = _training_accounting(sft, TrainingMode.SFT)
    assert result["model_generated_training_tokens_total"] == 0
    assert result["teacher_queried_positions_total"] is None
    assert result["teacher_queried_position_ratio"] is None

    empty = _training_accounting(
        _metrics(tmp_path / "empty", []),
        TrainingMode.OPD,
    )
    assert empty["total_trajectories"] == 0
    assert empty["selected_training_tokens_total"] == 0
    assert empty["selected_position_ratio"] is None


@pytest.mark.parametrize(
    "path",
    [
        "benchmarks/results/rtx4080-calc-hard-matched.json",
        "benchmarks/results/cpu-toy-calc-matched.json",
    ],
)
def test_committed_v1_results_remain_readable(path: str) -> None:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    result = BenchmarkResult.model_validate(payload)
    assert result.schema_version == 1
    assert result.arms
    assert (
        result.arms[0].selected_training_tokens_total
        == payload["arms"][0]["selected_training_tokens"]
    )
