"""RecoveryBench schema-v3 provenance and compatibility contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from miniverl.evaluation.benchmark import run_benchmark
from miniverl.evaluation.schema import (
    RECOVERY_BENCHMARK_SCHEMA_VERSION,
    ArmResult,
    BenchmarkConfig,
    BenchmarkResult,
    recovery_json_schema,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _arm(**updates: Any) -> ArmResult:
    payload: dict[str, Any] = {
        "name": "strict-opd-fresh",
        "mode": "opd",
        "seed": 1234,
        "run_id": "recovery-s1234",
        "run_dir": "recovery-s1234",
        "optimizer_steps": 8,
        "policy_version": 8,
        "tasks": 128,
        "success_rate": 0.5,
        "avg_turns": 4.0,
        "avg_tool_calls": 3.0,
        "invalid_tool_call_rate": 0.0,
        "generated_tokens_per_task": 96.0,
        "wall_seconds": 60.0,
        "recovery_metrics": {
            "recovery_after_error_rate": 0.75,
            "success_given_first_query_error": 0.70,
            "valid_sql_execution_rate": 0.90,
            "turns_to_recovery": 2.0,
            "subsets": {"controlled_intervention": {"tasks": 43}},
        },
        "stop_criterion": {"kind": "optimizer_steps", "target": 8},
        "overshoot": {"optimizer_steps": 0},
        "task_level_artifact": {
            "path": "release://recoverybench-v1-task-results.jsonl",
            "sha256": "a" * 64,
            "bytes": 1234,
        },
    }
    payload.update(updates)
    return ArmResult.model_validate(payload)


def _v3_result(**updates: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 3,
        "miniverl_version": "0.3.0",
        "name": "recoverybench-v1-equal-updates",
        "created_at": "2026-08-01T00:00:00Z",
        "invocation": ["miniverl", "benchmark", "recoverybench.yaml"],
        "budget_axis": "optimizer_steps",
        "cold_start": {"cycles": 24},
        "common_resolved_config": {"environment": {"name": "sqlite_recovery"}},
        "common_resolved_config_digest": "b" * 64,
        "preregistration_sha": "04fb44b9890596ea6e7ec0527e150c805b784798",
        "preregistration_digest": "c" * 64,
        "hypothesis_ids": ["H1", "H2", "H3"],
        "task_schedule_digest": "d" * 64,
        "template_registry_version": 1,
        "template_registry_digest": "e" * 64,
        "selected_teacher_candidate": {
            "id": "sqlite-recovery-candidate-a",
            "adapter_revision": "f" * 40,
        },
        "teacher_gate_results": [{"candidate": "candidate-a", "passed": True}],
        "teacher_preparation_cost": {
            "one_shot_seconds": 600.0,
            "amortized_over_5_students_seconds": 120.0,
            "amortized_over_10_students_seconds": 60.0,
        },
        "frozen_dataset_identity": {"1234": {"dataset_digest": "1" * 64}},
        "budget_view": "equal_optimizer_updates",
        "stop_criterion": {"kind": "optimizer_steps", "target": 8},
        "overshoot": {"maximum_optimizer_steps": 0},
        "recovery_metrics": {"analysis": "task_level_paired"},
        "task_level_artifacts": [
            {
                "path": "release://recoverybench-v1-task-results.jsonl",
                "sha256": "a" * 64,
                "bytes": 1234,
            }
        ],
        "result_analysis_version": "recoverybench-analysis-v1",
        "invalidation_status": {"valid": True, "reasons": []},
        "arms": [_arm().model_dump(mode="json")],
        "seeds": [1234, 20260727, 20260801],
    }
    payload.update(updates)
    return payload


def test_schema_v3_requires_recoverybench_provenance() -> None:
    assert RECOVERY_BENCHMARK_SCHEMA_VERSION == 3
    with pytest.raises(ValidationError, match="schema v3 requires provenance fields"):
        BenchmarkResult.model_validate(
            _v3_result(preregistration_sha=None, task_level_artifacts=[])
        )


def test_schema_v3_round_trips_all_declared_provenance() -> None:
    result = BenchmarkResult.model_validate(_v3_result())
    restored = BenchmarkResult.model_validate_json(result.model_dump_json())

    assert restored.schema_version == 3
    assert restored.arms[0].recovery_metrics["recovery_after_error_rate"] == 0.75
    assert restored.arms[0].task_level_artifact["sha256"] == "a" * 64
    assert restored.invalidation_status == {"valid": True, "reasons": []}


def test_committed_recovery_schema_matches_the_versioned_model() -> None:
    committed = json.loads(
        (REPO_ROOT / "benchmarks/schema/recoverybench-result.schema.json").read_text(
            encoding="utf-8"
        )
    )
    assert committed == recovery_json_schema()
    assert committed["$id"].endswith("/recoverybench-result.schema.json")


def test_schema_v3_config_requires_static_preflight_provenance() -> None:
    with pytest.raises(ValidationError, match="benchmark config schema v3 requires"):
        BenchmarkConfig.model_validate(
            {
                "schema_version": 3,
                "name": "incomplete-recoverybench",
                "base": {},
                "arms": [{"name": "cold-start-only"}],
            }
        )


@pytest.mark.torch
def test_schema_v3_benchmark_binds_runtime_task_artifact(tmp_path: Path) -> None:
    base = {
        "models": {
            "student": {"model_id": "toy-student"},
            "teacher": {"model_id": "toy-teacher"},
        },
        "environment": {
            "name": "calculator",
            "difficulty": "easy",
            "train_tasks": 1,
            "eval_tasks": 1,
            "test_tasks": 1,
        },
        "eval": {"enabled": False, "tasks": 1, "split": "test"},
    }
    metadata = _v3_result()
    spec = BenchmarkConfig.model_validate(
        {
            "schema_version": 3,
            "name": "tiny-recovery-v3",
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
            **{
                name: metadata[name]
                for name in (
                    "preregistration_sha",
                    "preregistration_digest",
                    "hypothesis_ids",
                    "task_schedule_digest",
                    "template_registry_version",
                    "template_registry_digest",
                    "selected_teacher_candidate",
                    "teacher_gate_results",
                    "teacher_preparation_cost",
                    "budget_view",
                    "stop_criterion",
                    "result_analysis_version",
                    "invalidation_status",
                )
            },
        }
    )

    result = run_benchmark(
        spec,
        output_dir=tmp_path / "benchmarks",
        invocation=["miniverl", "benchmark", "tiny-recovery.yaml"],
    )

    assert result.schema_version == 3
    assert result.preregistration_sha == metadata["preregistration_sha"]
    assert len(result.task_level_artifacts) == 1
    assert result.task_level_artifacts[0]["sha256"]
    assert result.arms[0].task_level_artifact == result.task_level_artifacts[0]
    assert result.invalidation_status == {"valid": True, "reasons": []}


@pytest.mark.parametrize(
    "path",
    [
        "benchmarks/results/gpu-calc-hard-equal-update-v2.json",
        "benchmarks/results/rtx4080-calc-hard-matched.json",
    ],
)
def test_schema_v3_reader_does_not_reinterpret_v1_or_v2(path: str) -> None:
    payload = json.loads((REPO_ROOT / path).read_text(encoding="utf-8"))
    result = BenchmarkResult.model_validate(payload)
    assert result.schema_version == payload["schema_version"]
