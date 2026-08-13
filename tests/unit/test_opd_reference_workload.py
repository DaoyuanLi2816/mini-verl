from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any


def _script() -> Any:
    path = Path("scripts/run_verl_opd_reference_workload.py")
    spec = importlib.util.spec_from_file_location("run_verl_opd_reference_workload", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resume_equivalence_excludes_only_the_run_specific_resolved_digest() -> None:
    script = _script()
    common = {
        "checkpoint_hashes": {
            "adapter.safetensors": "a",
            "optimizer.safetensors": "b",
            "state.json": "different-by-design",
        },
        "checkpoint_state": {
            "global_step": 8,
            "task_cursor": 32,
            "resolved_config_digest": "run-specific-a",
        },
        "trajectory_sha256": "c",
    }
    resumed = {
        **common,
        "checkpoint_state": {
            **common["checkpoint_state"],
            "resolved_config_digest": "run-specific-b",
        },
    }

    report = script._equivalence(common, resumed)

    assert report["status"] == "exact_match"
    assert report["adapter_and_optimizer_byte_identical"] is True
    assert report["training_state_fields_identical"] is True
    assert report["excluded_run_identity_field"] == "resolved_config_digest"


def test_measured_reference_workload_is_scoped_complete_and_frozen() -> None:
    path = Path("benchmarks/results/rtx4080-verl-opd-developer-v1.json")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "d13153734c20a084171763820a961a2c08511ded99854a28f1f5f169a843acf2"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["recipe"]["dataset_rows"] == 64
    assert payload["recipe"]["distinct_dataset_prompts"] == 64
    assert payload["recipe"]["distinct_prompts_consumed"] == 32
    assert payload["recipe"]["response_limit"] == 64
    assert payload["recipe"]["logical_batch"] == 4
    assert payload["recipe"]["optimizer_updates"] == 8
    assert payload["measurements"]["peak_reserved_gib"] <= 14.5
    assert payload["measurements"]["batch_downshifts"] == {
        "rollout_oom": 0,
        "update_chunk_oom": 0,
    }
    assert payload["resume"]["status"] == "exact_match"
    assert payload["verl"]["distributed_execution_tested"] is False
    assert payload["scientific_scope"]["alignment_quality_evaluated"] is False
    assert payload["scientific_scope"]["task_quality_evaluated"] is False


def test_reference_workload_figure_is_exactly_generated() -> None:
    publisher_path = Path("scripts/publish_verl_opd_reference_artifacts.py")
    spec = importlib.util.spec_from_file_location(
        "publish_verl_opd_reference_artifacts", publisher_path
    )
    assert spec is not None and spec.loader is not None
    publisher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(publisher)
    payload = publisher._load(Path("benchmarks/results/rtx4080-verl-opd-developer-v1.json"))

    expected = publisher.render(payload)
    expected_mobile = publisher.render_mobile(payload)
    actual = Path("docs/verl-opd-reference-workload.svg").read_text(encoding="utf-8")
    actual_mobile = Path("docs/verl-opd-reference-workload-mobile.svg").read_text(encoding="utf-8")

    assert actual == expected
    assert actual_mobile == expected_mobile
    assert "No task quality was evaluated" in actual
    assert "No task quality was evaluated" in actual_mobile
    assert "0 OOM downshifts" in actual
    assert "<title" in actual and "<desc" in actual


def test_second_family_smoke_is_compatibility_only_and_frozen() -> None:
    path = Path("benchmarks/results/rtx4080-smollm2-opd-family-smoke-v1.json")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "a6b7421de81af1afa0dd2a8350a0a66e649358cfc6c19da5a0993e625280685e"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["tokenizer_identity"]["student_teacher_match"] is True
    assert payload["workload"]["optimizer_updates"] == 1
    assert payload["artifacts"]["peft_load_verified"] is True
    assert payload["scope"]["rollout_completed"] is True
    assert payload["scope"]["teacher_scoring_completed"] is True
    assert payload["scope"]["optimizer_update_completed"] is True
    assert payload["scope"]["task_quality_evaluated"] is False
    assert payload["scope"]["alignment_quality_evaluated"] is False
    assert payload["scope"]["full_recipe_supported"] is False
    assert payload["runtime"]["distributed_execution_tested"] is False
