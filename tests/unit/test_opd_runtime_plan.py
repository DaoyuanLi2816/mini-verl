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
    assert plan.acknowledgement_required_mappings
    assert all(
        item["reason"] and item["local_target"] for item in plan.acknowledgement_required_mappings
    )


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


def test_direct_gkd_can_bind_the_measured_vllm_runtime_without_changing_profile_identity() -> None:
    compiled = load_verl_opd_v08_source("builtin:qwen3-0.6b-1.7b-opd")
    default = compile_native_run_config(compiled)
    external = compile_native_run_config(compiled, rollout_backend="vllm")

    assert default.run.profile_identity == external.run.profile_identity
    assert default.rollout.backend.value == "hf_reference"
    assert external.rollout.backend.value == "vllm"
    assert external.rollout.record_logprobs is False
    assert external.rollout.engine.managed is True
    assert external.rollout.engine.host == "127.0.0.1"
    assert external.rollout.engine.memory_fraction == 0.5


def test_runtime_critical_model_settings_are_explicit_and_effective() -> None:
    source = load_verl_opd_v08_source("builtin:qwen3-0.6b-1.7b-opd").source.model_dump(mode="json")
    source["miniverl"]["actor_runtime"]["dtype"] = "float16"
    source["miniverl"]["actor_runtime"]["quantization"] = "int8"
    source["miniverl"]["actor_runtime"]["attn_implementation"] = "eager"
    source["distillation"]["teacher_models"]["teacher_model"]["inference"]["dtype"] = "float32"
    source["miniverl"]["teacher_runtime"]["quantization"] = "none"
    source["miniverl"]["teacher_runtime"]["attn_implementation"] = "eager"

    compiled = compile_verl_opd_v08(source)
    native = compile_native_run_config(compiled)
    plan = build_system_plan(compiled)

    assert native.models.student.dtype.value == "float16"
    assert native.models.student.quantization.value == "int8"
    assert native.models.student.attn_implementation == "eager"
    assert native.models.teacher.dtype.value == "float32"
    assert native.models.teacher.quantization.value == "none"
    assert native.models.teacher.attn_implementation == "eager"
    assert plan.student["runtime"] == {
        "dtype": "float16",
        "quantization": "int8",
        "attn_implementation": "eager",
    }
    assert plan.teacher["runtime"] == {
        "dtype": "float32",
        "quantization": "none",
        "attn_implementation": "eager",
    }


def test_direct_gkd_batch_and_token_budgets_have_distinct_effects() -> None:
    source = load_verl_opd_v08_source("builtin:qwen3-0.6b-1.7b-opd").source.model_dump(mode="json")
    source["data"]["train_batch_size"] = 4
    source["actor_rollout_ref"]["actor"]["ppo_mini_batch_size"] = 1
    source["actor_rollout_ref"]["actor"]["ppo_max_token_len_per_gpu"] = 777
    source["actor_rollout_ref"]["rollout"]["max_num_batched_tokens"] = 333
    source["miniverl"]["batching"]["update_trajectory_batch_size"] = 2

    compiled = compile_verl_opd_v08(source)
    native = compile_native_run_config(compiled)
    entries = {item.upstream_field: item for item in compiled.compatibility}

    assert entries["actor_rollout_ref.actor.ppo_mini_batch_size"].classification == (
        "informational_only"
    )
    assert entries["actor_rollout_ref.actor.ppo_mini_batch_size"].local_target is None
    assert native.train.rollouts_per_cycle == 4
    assert native.train.gradient_accumulation_steps == 4
    assert native.train.trajectory_batch_size == 2
    assert native.train.max_update_padded_tokens == 777
    assert native.rollout.max_padded_tokens == 333


def test_existing_student_adapter_is_compiled_as_a_pinned_trainable_input() -> None:
    source = load_verl_opd_v08_source("builtin:qwen3-0.6b-1.7b-opd").source.model_dump(mode="json")
    source["actor_rollout_ref"]["model"]["lora_adapter_path"] = "owner/student-adapter"
    source["miniverl"]["student_adapter"] = {
        "source": "hub",
        "revision": "1" * 40,
        "base_model_revision": source["miniverl"]["student_revision"],
        "tokenizer_fingerprint": "a" * 64,
    }

    compiled = compile_verl_opd_v08(source)
    native = compile_native_run_config(compiled)
    adapter = native.models.student.adapter

    assert adapter is not None
    assert adapter.path == "owner/student-adapter"
    assert adapter.source.value == "hub"
    assert adapter.revision == "1" * 40
    assert adapter.base_model_revision == source["miniverl"]["student_revision"]
    assert adapter.tokenizer_fingerprint == "a" * 64


def test_student_adapter_requires_complete_pinned_provenance() -> None:
    source = load_verl_opd_v08_source("builtin:qwen3-0.6b-1.7b-opd").source.model_dump(mode="json")
    source["actor_rollout_ref"]["model"]["lora_adapter_path"] = "owner/student-adapter"

    with pytest.raises(ConfigError, match="student_adapter"):
        compile_verl_opd_v08(source)


def test_shared_backbone_fails_for_distinct_model_pair() -> None:
    compiled = load_verl_opd_v08_source(
        "builtin:qwen3-0.6b-1.7b-opd", overrides=["miniverl.runtime.mode=shared_backbone"]
    )
    with pytest.raises(ConfigError, match="same base model"):
        build_system_plan(compiled)


def test_unknown_quantized_model_sizes_require_a_probe_instead_of_illegal_swap(
    tmp_path: Path,
) -> None:
    source = load_verl_opd_v08_source("builtin:qwen3-0.6b-1.7b-opd").source.model_dump(mode="json")
    source["actor_rollout_ref"]["model"]["path"] = "owner/student"
    source["distillation"]["teacher_models"]["teacher_model"]["model_path"] = "owner/teacher"
    compiled = compile_verl_opd_v08(source, require_executable=False)
    plan = build_system_plan(compiled)
    assert plan.executable is False
    assert plan.local_execution["strategy"] == "requires_probe"
    assert plan.local_execution["placement_not_proven"] is True
    assert plan.local_execution["executable_without_probe"] is False
    assert "unknown" in plan.local_execution["reason"]
    with pytest.raises(ConfigError, match="probe"):
        compile_native_run_config(compiled, system_plan=plan)
