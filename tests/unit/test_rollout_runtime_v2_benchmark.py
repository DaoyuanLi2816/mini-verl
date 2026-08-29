"""Preregistration and result contracts for Rollout Runtime v2."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jsonschema
import yaml

ROOT = Path(__file__).resolve().parents[2]
PREREGISTRATION = ROOT / "benchmarks/preregistration/rollout-runtime-v2.yaml"
SCHEMA = ROOT / "benchmarks/schema/rollout-runtime-v2.schema.json"
FROZEN = ROOT / "benchmarks/results/gpu-calc-hard-equal-update-v2.json"


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
