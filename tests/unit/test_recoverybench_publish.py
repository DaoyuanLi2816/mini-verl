"""Frozen RecoveryBench publication and analysis contracts."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

from miniverl.evaluation.benchmark import resolve_benchmark_configs
from miniverl.evaluation.schema import BenchmarkConfig, BenchmarkResult

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
    expected_digest = "9c4c2ec19a56cebb2b2c1c0f3c7e504a9285467c99ae1590488251fbf2ff3934"
    assert hashlib.sha256(preregistration.read_bytes()).hexdigest() == expected_digest

    expected = {
        "recoverybench_v1_equal_updates.yaml": ("equal_optimizer_updates", 8),
        "recoverybench_v1_equal_selected_tokens.yaml": (
            "equal_selected_training_tokens",
            6224,
        ),
        "recoverybench_v1_equal_wall_time.yaml": ("equal_gpu_wall_time", 50),
    }
    for name, (view, target) in expected.items():
        config = BenchmarkConfig.from_yaml(ROOT / "benchmarks/configs" / name)
        assert config.preregistration_digest == expected_digest
        assert config.preregistration_sha == "7087b3a333463b88a62ffed73daee2c85d039145"
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


def test_published_recoverybench_artifacts_are_exact_and_data_bound() -> None:
    expected = {
        "benchmarks/results/recoverybench-v1-equal-updates.json": (
            "6ce2e6837e12b99ebc4fad6d27ce3e69c92e295ff3b9b60e0f68c2d308022384"
        ),
        "benchmarks/results/recoverybench-v1-equal-selected-tokens.json": (
            "fe4c9afc799724dfe7a32e631676a1e5177c44559a7374d2ea31da135354f137"
        ),
        "benchmarks/results/recoverybench-v1-equal-wall-time.json": (
            "425b0fa568f37b09e61af731d3da5009bd3833bddde6efaf2c66e9dba8355cbe"
        ),
        "benchmarks/results/recoverybench-v1-task-results.jsonl": (
            "aff96bffc6da27240a852410ac041bd4d95badf34cad030e6f437be1491a55ad"
        ),
        "benchmarks/results/recoverybench-v1-analysis.json": (
            "8a6891f74aed80f07ec00d5ea1909895c579346e1abbb1d5d95a354bb46c6b81"
        ),
        "docs/recoverybench/recovery-success.svg": (
            "0deab77a739cb27bd76f7399297231ebe7bf323a04026d43a1b5af78ace42dad"
        ),
        "docs/recoverybench/cost-quality-pareto.svg": (
            "865725bded3982ecf3bf0e3582342cf23e123a8d50269e8779336a54c453afcc"
        ),
        "docs/recoverybench/fresh-vs-frozen.svg": (
            "54ce1275ce1f828eb22ec2f518b227f7bf8a175bfcea93c8b8e3a063e0f05897"
        ),
        "paper/recoverybench-v1/recoverybench-v1.pdf": (
            "c506300599942445f24b30a4e0d7e01972c75daa7834f6d1eff5b8132dce93af"
        ),
    }
    for relative, digest in expected.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest

    result_paths = {
        "equal_optimizer_updates": ROOT / "benchmarks/results/recoverybench-v1-equal-updates.json",
        "equal_selected_training_tokens": ROOT
        / "benchmarks/results/recoverybench-v1-equal-selected-tokens.json",
        "equal_gpu_wall_time": ROOT / "benchmarks/results/recoverybench-v1-equal-wall-time.json",
    }
    results = {
        view: BenchmarkResult.model_validate_json(path.read_text(encoding="utf-8"))
        for view, path in result_paths.items()
    }
    assert all(result.schema_version == 3 for result in results.values())
    assert all(
        result.invalidation_status == {"valid": True, "reasons": []} for result in results.values()
    )

    analysis = json.loads(
        (ROOT / "benchmarks/results/recoverybench-v1-analysis.json").read_text(encoding="utf-8")
    )
    assert analysis["source_result_sha256"] == {
        view: hashlib.sha256(path.read_bytes()).hexdigest() for view, path in result_paths.items()
    }
    rendered = _script().render_figures(
        results["equal_optimizer_updates"],
        analysis,
        analysis["source_result_sha256"]["equal_optimizer_updates"],
    )
    for name, content in rendered.items():
        assert (ROOT / "docs/recoverybench" / name).read_text(encoding="utf-8") == content
