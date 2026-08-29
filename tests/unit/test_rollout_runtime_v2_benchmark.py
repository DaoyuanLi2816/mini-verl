"""Preregistration and result contracts for Rollout Runtime v2."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import jsonschema
import yaml

ROOT = Path(__file__).resolve().parents[2]
PREREGISTRATION = ROOT / "benchmarks/preregistration/rollout-runtime-v2.yaml"
SCHEMA = ROOT / "benchmarks/schema/rollout-runtime-v2.schema.json"
FROZEN = ROOT / "benchmarks/results/gpu-calc-hard-equal-update-v2.json"
SELECTED_SCHEMA = ROOT / "benchmarks/schema/rollout-runtime-v2-selected.schema.json"
RESULT_SCHEMA = ROOT / "benchmarks/schema/rollout-runtime-v2-result.schema.json"
HF_CACHED = ROOT / "benchmarks/evidence/rollout-runtime-v2/hf-cached-rtx4080-raw.json"
VLLM = ROOT / "benchmarks/evidence/rollout-runtime-v2/vllm-rtx4080-raw.json"
SELECTED_RESULT = ROOT / "benchmarks/results/rollout-runtime-v2-rtx4080.json"


def test_rollout_runtime_v2_workload_is_preregistered_before_baseline() -> None:
    payload = yaml.safe_load(PREREGISTRATION.read_text(encoding="utf-8"))

    assert payload["status"] == "preregistered_before_baseline_measurement"
    assert payload["models"]["actor"]["revision"] == ("c1899de289a04d12100db370d81485cdf75e47ca")
    assert payload["models"]["teacher"]["revision"] == ("70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
    assert payload["workload"]["prompt_lengths"] == [128, 512]
    assert payload["workload"]["response_bounds"] == [64, 256, 512]
    assert payload["workload"]["samples_per_prompt"] == [1, 4]
    assert {row["name"] for row in payload["workload"]["sampling"]} == {
        "greedy",
        "seeded_stochastic",
    }
    assert payload["performance_gates"]["hf_cached_speedup_over_hf_reference"] == {
        "response_256": 2.0,
        "response_512": 2.0,
    }


def test_rollout_runtime_v2_preserves_frozen_calculator() -> None:
    assert hashlib.sha256(FROZEN.read_bytes()).hexdigest() == (
        "53fc1d4d5b7adee09618d77ad62d4086ba56b78569832d6fc7c3bcd5c2695bbc"
    )


def test_measured_hf_reference_baseline_is_exact_and_schema_valid() -> None:
    expected = {
        "benchmarks/preregistration/rollout-runtime-v2.yaml": (
            "8cc3ba738c69b59ed19c22c1de874fd00249404198a3e05983477dc8899bb7e5"
        ),
        "benchmarks/evidence/rollout-runtime-v2/hf-reference-raw.json": (
            "2e303eabb559b843d25377a7c72e0aeb0219eec7eb8bc109c0419839b3251170"
        ),
        "benchmarks/results/rollout-runtime-v2-hf-reference.json": (
            "b25daee7ee726b7a7be18d7dbe26590fb61325bf212cfd9e7110b69f5fa8889c"
        ),
    }
    for relative, digest in expected.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest

    payload = json.loads(
        (ROOT / "benchmarks/results/rollout-runtime-v2-hf-reference.json").read_text(
            encoding="utf-8"
        )
    )
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    jsonschema.validate(payload, schema)
    assert payload["measurement_status"] == "measured_baseline"
    assert payload["failures"] == []
    assert len(payload["cells"]) == 24
    assert all(cell["status"] == "completed" for cell in payload["cells"])
    assert payload["source"] == {
        "commit": "0d6e0070ae73ef35f718aec3624ee5263ac96e3a",
        "dirty": False,
        "miniverl_version": "0.11.0.dev0",
        "wheel_sha256": "0256bd9e63ca6ed52999a5073a3577a008581a8d4418062314750d86f21cd5fe",
    }
    assert payload["policy_identity"]["base_weight_digest_sha256"] == (
        "b29cd98b83f9bddc7ec8943be5f142243e956f448f13456847849e5b8615b413"
    )
    assert max(cell["memory"]["peak_reserved_bytes"] for cell in payload["cells"]) == (1166016512)
    raw_payload = json.loads(
        (ROOT / "benchmarks/evidence/rollout-runtime-v2/hf-reference-raw.json").read_text(
            encoding="utf-8"
        )
    )
    measurement_view = dict(payload)
    measurement_view.pop("raw_measurement_sha256")
    measurement_view.pop("policy_identity")
    measurement_view["environment"] = dict(measurement_view["environment"])
    measurement_view["environment"].pop("driver_version")
    assert measurement_view == raw_payload
    for cell in payload["cells"]:
        assert cell["counts"]["generated_tokens"] == (
            4 * cell["samples_per_prompt"] * cell["response_bound"]
        )
        for phase in ("prefill", "decode", "teacher_scoring", "actor_update", "full_cycle"):
            assert cell["phases"][phase]["status"] == "not_measured"
            assert "median_seconds" not in cell["phases"][phase]


def _phase(status: str, measured: bool = False) -> dict[str, object]:
    payload: dict[str, object] = {"status": status}
    if measured:
        payload.update({"median_seconds": 1.0, "minimum_seconds": 0.9})
    return payload


def test_rollout_runtime_v2_schema_keeps_unavailable_phases_distinct_from_zero() -> None:
    phases = {
        "cold_start": _phase("measured", measured=True),
        "prefill": _phase("not_measured"),
        "decode": _phase("not_measured"),
        "rollout_total": _phase("measured", measured=True),
        "policy_sync": _phase("not_applicable"),
        "teacher_scoring": _phase("not_measured"),
        "actor_update": _phase("not_measured"),
        "full_cycle": _phase("not_measured"),
        "teardown": _phase("measured", measured=True),
    }
    result = {
        "schema_version": 1,
        "name": "rollout-runtime-v2-hf-reference",
        "measurement_status": "measured_baseline",
        "source": {
            "commit": "a" * 40,
            "dirty": False,
            "miniverl_version": "0.11.0.dev0",
            "wheel_sha256": "b" * 64,
        },
        "raw_measurement_sha256": "9" * 64,
        "preregistration_sha256": "c" * 64,
        "workload_manifest_sha256": "d" * 64,
        "frozen_calculator_sha256": (
            "53fc1d4d5b7adee09618d77ad62d4086ba56b78569832d6fc7c3bcd5c2695bbc"
        ),
        "environment": {},
        "models": {
            "actor": {
                "id": "Qwen/Qwen3-0.6B",
                "revision": "e" * 40,
                "tokenizer_revision": "e" * 40,
            },
            "teacher": {
                "id": "Qwen/Qwen3-1.7B",
                "revision": "f" * 40,
                "tokenizer_revision": "f" * 40,
            },
            "dtype": "bfloat16",
            "quantization": "nf4",
        },
        "backend": {
            "name": "hf_reference",
            "version": "5.14.1",
            "reproducibility_class": "same_process_seeded_reference",
        },
        "policy_identity": {
            "policy_version": 0,
            "profile_identity": "rollout-runtime-v2-hf-reference-baseline-v1",
            "base_revision": "e" * 40,
            "base_weight_digest_sha256": "8" * 64,
            "adapter_digest_sha256": None,
            "identity_digest_sha256": "7" * 64,
        },
        "cells": [
            {
                "cell_id": "p128-r64-n1-greedy",
                "prompt_length": 128,
                "response_bound": 64,
                "samples_per_prompt": 1,
                "sampling": "greedy",
                "status": "completed",
                "counts": {
                    "logical_prompts": 4,
                    "generated_trajectories": 4,
                    "prompt_tokens": 512,
                    "generated_tokens": 256,
                    "physical_batches": 1,
                },
                "phases": phases,
                "rates": {
                    "time_to_first_token_seconds": None,
                    "prompt_tokens_per_second": 512.0,
                    "output_tokens_per_second": 256.0,
                },
                "memory": {
                    "peak_allocated_bytes": 1,
                    "peak_reserved_bytes": 2,
                    "process_rss_bytes": 3,
                    "teardown_residual_allocated_bytes": 0,
                },
                "output_token_ids_sha256": "1" * 64,
                "sampled_logprobs_sha256": "2" * 64,
                "oom_downshifts": 0,
                "error": None,
            }
        ],
        "failures": [],
    }

    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    jsonschema.validate(result, schema)
    for name in ("prefill", "decode", "teacher_scoring", "actor_update", "full_cycle"):
        assert "median_seconds" not in phases[name]


def test_selected_backend_evidence_is_exact_private_path_free_and_schema_valid() -> None:
    expected = {
        HF_CACHED: "a9f8f0b0275d940f30497a0d88d76da0e112ffbe2bc1b37a42c6ed313b852242",
        VLLM: "abf14d73a7289f9619515943b4c8d7dafe7cd187fefe928655739ac7ed1ab16c",
        SELECTED_RESULT: "32be86856263ccdf787986c4ec54570d323af8c238dc1664e00a9ce41ca393c4",
    }
    raw_schema = json.loads(SELECTED_SCHEMA.read_text(encoding="utf-8"))
    result_schema = json.loads(RESULT_SCHEMA.read_text(encoding="utf-8"))
    for path, digest in expected.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        text = path.read_text(encoding="utf-8")
        for private_fragment in (
            "14191",
            "daoyuanli/.venvs",
            "OneDrive",
            "AppData",
            "C:\\\\Users",
        ):
            assert private_fragment not in text
    for path in (HF_CACHED, VLLM):
        jsonschema.validate(json.loads(path.read_text(encoding="utf-8")), raw_schema)
    jsonschema.validate(json.loads(SELECTED_RESULT.read_text(encoding="utf-8")), result_schema)


def test_selected_backend_result_preserves_failed_release_gate() -> None:
    result = json.loads(SELECTED_RESULT.read_text(encoding="utf-8"))
    assert len(result["cells"]) == 24
    assert result["source"] == {
        "commit": "690db9b079bfecfec14dc3bedc3aa0308cbacf60",
        "dirty": False,
        "miniverl_version": "0.11.0.dev0",
        "wheel_sha256": "266fcd59bb3e85b02974a92b26c9a4e59c7c9b9205853d17baadcfffdb898e27",
    }
    assert result["gates"]["hf_cached_speedup_over_hf_reference"]["passed"] is False
    assert result["gates"]["selected_external_engine_advantage_over_hf_cached"]["passed"] is True
    assert result["gates"]["vllm_policy_refresh"]["passed"] is True
    assert result["gates"]["vllm_pg_logprob_conformance"]["passed"] is False
    assert result["release_decision"] == {
        "action": "publish evidence, optimize hf_cached, rerun the frozen workload",
        "blocking_gate": "hf_cached_speedup_over_hf_reference",
        "v0_11_0_publishable": False,
    }


def test_selected_backend_aggregate_is_reproducible() -> None:
    subprocess.run(
        [sys.executable, "scripts/publish_rollout_runtime_v2_evidence.py", "--check"],
        cwd=ROOT,
        check=True,
    )
