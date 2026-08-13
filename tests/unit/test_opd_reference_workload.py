from __future__ import annotations

import importlib.util
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
