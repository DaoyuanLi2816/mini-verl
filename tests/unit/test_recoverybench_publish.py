"""Frozen RecoveryBench publication and analysis contracts."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from miniverl.evaluation.benchmark import resolve_benchmark_configs
from miniverl.evaluation.schema import ArmResult, BenchmarkConfig, BenchmarkResult

ROOT = Path(__file__).resolve().parents[2]


def _script() -> ModuleType:
    path = ROOT / "scripts" / "publish_recoverybench_artifacts.py"
    spec = importlib.util.spec_from_file_location("publish_recoverybench_artifacts", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_final_recoverybench_configs_are_frozen_and_preflight_clean() -> None:
    preregistration = ROOT / "benchmarks/preregistration/recoverybench-v1.yaml"
    revision_13_digest = "9c4c2ec19a56cebb2b2c1c0f3c7e504a9285467c99ae1590488251fbf2ff3934"
    revision_14_digest = "d3c7c352ade05e6be2f7e70db29e0485a180266a0dc85be4e73bbdde6084cd87"
    assert hashlib.sha256(preregistration.read_bytes()).hexdigest() == revision_14_digest
    amendment = ROOT / "benchmarks/preregistration/recoverybench-v1-wall-time-amendment.json"
    assert hashlib.sha256(amendment.read_bytes()).hexdigest() == (
        "3b9eff041e6718eb56f10438fe6f3ccb7a8b1f140a9bd6e5b9e98ef64e5b3256"
    )

    expected = {
        "recoverybench_v1_equal_updates.yaml": (
            "equal_optimizer_updates",
            8,
            revision_13_digest,
            "7087b3a333463b88a62ffed73daee2c85d039145",
        ),
        "recoverybench_v1_equal_selected_tokens.yaml": (
            "equal_selected_training_tokens",
            6224,
            revision_13_digest,
            "7087b3a333463b88a62ffed73daee2c85d039145",
        ),
        "recoverybench_v1_equal_wall_time.yaml": (
            "equal_gpu_wall_time",
            24,
            revision_14_digest,
            "6f4fabbd74cbb5af9c5427fa9d4fcc0d3e9752e7",
        ),
    }
    for name, (view, target, digest, preregistration_sha) in expected.items():
        config = BenchmarkConfig.from_yaml(ROOT / "benchmarks/configs" / name)
        assert config.preregistration_digest == digest
        assert config.preregistration_sha == preregistration_sha
        assert config.eval_split == "test"
        assert config.seeds == [1234, 20260727, 20260801]
        assert config.budget_view == view
        assert target in config.stop_criterion.values()
        for seed in config.seeds:
            resolve_benchmark_configs(config, seed=seed)

    primary = BenchmarkConfig.from_yaml(
        ROOT / "benchmarks/configs/recoverybench_v1_equal_updates.yaml"
    )
    oracle = next(arm for arm in primary.arms if arm.name == "offline-kd-oracle")
    assert oracle.overrides["offline_kd"]["collection_tasks"] == 64


def test_paired_bootstrap_is_deterministic_and_svg_never_has_negative_width() -> None:
    script = _script()
    first = script._paired_interval([-1.0, 0.0, 1.0])
    second = script._paired_interval([-1.0, 0.0, 1.0])
    assert first == second
    assert first["replicates"] == 10_000

    svg = script._svg("title", "subtitle", [("negative", -0.25, 0.5)], x_label="rate")
    assert 'width="-' not in svg
    assert "-0.250" in svg


def test_wall_time_publication_rejects_a_cycle_capped_arm() -> None:
    script = _script()
    result = BenchmarkResult.model_construct(
        budget_view="equal_gpu_wall_time",
        arms=[
            ArmResult.model_construct(
                name="continued-sft",
                seed=1234,
                stop_criterion={"kind": "configured_cycles", "target": 8, "actual": 8},
                overshoot={
                    "axis": "optimizer_steps",
                    "target": 8,
                    "actual": 8,
                    "value": 0,
                },
            )
        ],
    )

    with pytest.raises(ValueError, match="wall_seconds"):
        script._validate_budget_contract(result, "equal_gpu_wall_time")
