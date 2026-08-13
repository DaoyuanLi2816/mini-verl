from __future__ import annotations

from pathlib import Path

import pytest

from miniverl.bridge.opd_runtime import build_system_plan, compile_native_run_config
from miniverl.bridge.opd_v08 import compile_verl_opd_v08, load_verl_opd_v08_source
from miniverl.errors import ConfigError


def test_builtin_plan_is_weight_free_truthful_and_executable() -> None:
    compiled = load_verl_opd_v08_source("builtin:qwen3-0.6b-1.7b-opd")
    plan = build_system_plan(compiled)

    assert plan.executable is True
    assert compiled.source.actor_rollout_ref.rollout.name == "vllm"
    assert compiled.source.distillation.teacher_models.teacher_model.inference.name == "vllm"
    engine_rules = {
        item.upstream_field: item
        for item in compiled.compatibility
        if item.upstream_field.endswith("name")
    }
    assert engine_rules["actor_rollout_ref.rollout.name"].classification == "locally_reinterpreted"
    assert (
        engine_rules["distillation.teacher_models.teacher_model.inference.name"].classification
        == "locally_reinterpreted"
    )
    assert plan.upstream["commit"] == "7aed6b230776f963fa09509c10d9c3a767d1102c"
    assert plan.student["revision"] == "c1899de289a04d12100db370d81485cdf75e47ca"
    assert plan.teacher["revision"] == "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    assert plan.memory["status"] == "estimated"
    assert plan.memory["measured_peak_reserved_gib"] is None
    assert plan.time_to_first_update == {"status": "unknown", "seconds": None}
    assert plan.local_execution["distributed_execution"] is False
    assert plan.unsupported_fields == []


def test_native_compilation_preserves_pure_opd_semantics() -> None:
    compiled = load_verl_opd_v08_source("builtin:qwen3-0.6b-1.7b-opd")
    native = compile_native_run_config(compiled)

    assert native.source.train_files == ["data/opd-smoke.parquet"]
    assert native.models.student.revision == "c1899de289a04d12100db370d81485cdf75e47ca"
    assert native.models.teacher.revision == "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    assert native.loss.mode.value == "forward_kl_topk"
    assert native.loss.aggregation.value == "token-mean"
    assert native.loss.sampled_token_nll_weight == 0.0
    assert native.train.cycles == 1
    assert native.eval.enabled is False


def test_shared_backbone_fails_for_distinct_model_pair() -> None:
    compiled = load_verl_opd_v08_source(
        "builtin:qwen3-0.6b-1.7b-opd", overrides=["miniverl.runtime.mode=shared_backbone"]
    )
    with pytest.raises(ConfigError, match="same base model"):
        build_system_plan(compiled)


def test_unknown_model_sizes_make_auto_fail_safe_to_swap(tmp_path: Path) -> None:
    source = load_verl_opd_v08_source("builtin:qwen3-0.6b-1.7b-opd").source.model_dump(mode="json")
    source["actor_rollout_ref"]["model"]["path"] = "owner/student"
    source["distillation"]["teacher_models"]["teacher_model"]["model_path"] = "owner/teacher"
    compiled = compile_verl_opd_v08(source)
    plan = build_system_plan(compiled)
    assert plan.local_execution["strategy"] == "swap"
    assert "unknown" in plan.local_execution["reason"]
