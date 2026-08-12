from __future__ import annotations

import json
from pathlib import Path


def test_rtx4080_runtime_record_is_scoped_and_complete() -> None:
    path = Path("benchmarks/results/rtx4080-verl-opd-runtime-v1.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["kind"] == "single_gpu_runtime_conformance"
    assert payload["status"] == "measured"
    assert payload["hardware"]["gpu_count"] == 1
    assert payload["verl"]["commit"] == "7aed6b230776f963fa09509c10d9c3a767d1102c"
    assert payload["verl"]["distributed_execution_tested"] is False
    assert payload["measurements"]["peak_reserved_gib"] <= 14.5
    assert payload["recipe"]["response_limit"] == 16
    assert payload["recipe"]["optimizer_updates"] == 1
    assert payload["artifacts"]["standard_peft_load_verified"] is True
    assert payload["scientific_scope"] == {
        "runtime_correctness_only": True,
        "alignment_quality_evaluated": False,
        "opd_beats_sft_dpo_or_kd_claimed": False,
    }


def test_unmeasured_hardware_rows_are_not_promoted_to_measurements() -> None:
    text = Path("docs/opd-quickstart.md").read_text(encoding="utf-8")
    assert "12 GiB CUDA GPU" in text and "not measured" in text
    assert "24 GiB CUDA GPU" in text and "not measured" in text
    assert "demonstrates\nruntime and artifact correctness only" in text
