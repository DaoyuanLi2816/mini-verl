#!/usr/bin/env python3
"""Validate and publish the v0.11.0 Rollout Runtime v2 candidate evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION = ROOT / "benchmarks/preregistration/rollout-runtime-v2.yaml"
BASELINE = ROOT / "benchmarks/results/rollout-runtime-v2-hf-reference.json"
HF_CACHED = ROOT / "benchmarks/evidence/rollout-runtime-v2/hf-cached-v2-rtx4080-candidate-raw.json"
VLLM = ROOT / "benchmarks/evidence/rollout-runtime-v2/vllm-cudagraph-rtx4080-candidate-raw.json"
OUTPUT = ROOT / "benchmarks/results/rollout-runtime-v2-v0.11.0-candidate-rtx4080.json"
FROZEN_CALCULATOR = ROOT / "benchmarks/results/gpu-calc-hard-equal-update-v2.json"
FROZEN_CALCULATOR_SHA256 = "53fc1d4d5b7adee09618d77ad62d4086ba56b78569832d6fc7c3bcd5c2695bbc"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")


def _expected_cells(preregistration: dict[str, Any]) -> set[str]:
    workload = preregistration["workload"]
    return {
        f"p{prompt}-r{response}-n{samples}-{sampling['name']}"
        for prompt in workload["prompt_lengths"]
        for response in workload["response_bounds"]
        for samples in workload["samples_per_prompt"]
        for sampling in workload["sampling"]
    }


def _cells(payload: dict[str, Any], expected: set[str]) -> dict[str, dict[str, Any]]:
    rows = payload.get("cells")
    if not isinstance(rows, list):
        raise ValueError("benchmark evidence has no cell list")
    by_id = {str(row["cell_id"]): row for row in rows}
    if len(by_id) != len(rows) or set(by_id) != expected:
        raise ValueError("benchmark evidence does not match the preregistered 24-cell matrix")
    return by_id


def _refresh_gate(payload: dict[str, Any]) -> dict[str, Any]:
    refresh = payload["refresh_probe"]
    passed = bool(
        len(refresh["cycles"]) == 8
        and refresh["all_syncs_confirmed"]
        and refresh["all_policy_identities_unique"]
        and not refresh["strictly_monotonic_memory_growth"]
    )
    return {
        "cycles": len(refresh["cycles"]),
        "all_syncs_confirmed": refresh["all_syncs_confirmed"],
        "all_policy_identities_unique": refresh["all_policy_identities_unique"],
        "strictly_monotonic_memory_growth": refresh["strictly_monotonic_memory_growth"],
        "passed": passed,
    }


def build_result() -> dict[str, Any]:
    preregistration = yaml.safe_load(PREREGISTRATION.read_text(encoding="utf-8"))
    baseline = _load(BASELINE)
    hf_cached = _load(HF_CACHED)
    vllm = _load(VLLM)
    expected = _expected_cells(preregistration)
    if len(expected) != 24:
        raise ValueError("the preregistered workload is no longer the frozen 24-cell matrix")
    if _sha256(FROZEN_CALCULATOR) != FROZEN_CALCULATOR_SHA256:
        raise ValueError("the immutable calculator benchmark changed")
    if hf_cached["source"] != vllm["source"]:
        raise ValueError("candidate backends were not measured from the same source and wheel")
    if hf_cached["environment"] != vllm["environment"]:
        raise ValueError("candidate backends were not measured in the same software environment")
    preregistration_sha256 = _sha256(PREREGISTRATION)
    for payload in (hf_cached, vllm):
        if payload["preregistration_sha256"] != preregistration_sha256:
            raise ValueError("candidate evidence does not bind the frozen preregistration")
        if payload["frozen_calculator_sha256"] != FROZEN_CALCULATOR_SHA256:
            raise ValueError("candidate evidence does not bind the frozen calculator result")

    baseline_cells = _cells(baseline, expected)
    hf_cells = _cells(hf_cached, expected)
    vllm_cells = _cells(vllm, expected)
    cells: list[dict[str, Any]] = []
    for cell_id in sorted(expected):
        baseline_cell = baseline_cells[cell_id]
        hf_cell = hf_cells[cell_id]
        vllm_cell = vllm_cells[cell_id]
        baseline_rate = float(baseline_cell["rates"]["output_tokens_per_second"])
        hf_rate = float(hf_cell["output_tokens_per_second"])
        vllm_rate = float(vllm_cell["output_tokens_per_second"])
        cells.append(
            {
                "cell_id": cell_id,
                "prompt_length": hf_cell["prompt_length"],
                "response_bound": hf_cell["response_bound"],
                "samples_per_prompt": hf_cell["samples_per_prompt"],
                "sampling": hf_cell["sampling"],
                "hf_reference_output_tokens_per_second": baseline_rate,
                "hf_cached_output_tokens_per_second": hf_rate,
                "vllm_output_tokens_per_second": vllm_rate,
                "hf_cached_speedup_over_hf_reference": hf_rate / baseline_rate,
                "vllm_speedup_over_hf_cached": vllm_rate / hf_rate,
                "hf_cached_peak_total_gpu_memory_mib": hf_cell["peak_total_gpu_memory_mib"],
                "vllm_peak_total_gpu_memory_mib": vllm_cell["peak_total_gpu_memory_mib"],
            }
        )

    hf_responses: dict[str, dict[str, Any]] = {}
    for response in (256, 512):
        values = [
            row["hf_cached_speedup_over_hf_reference"]
            for row in cells
            if row["response_bound"] == response
        ]
        hf_responses[str(response)] = {
            "required_minimum_speedup": 2.0,
            "observed_minimum_speedup": min(values),
            "observed_maximum_speedup": max(values),
            "passed": all(value >= 2.0 for value in values),
        }
    external_values = [
        row["vllm_speedup_over_hf_cached"] for row in cells if row["response_bound"] in (256, 512)
    ]
    memory_limit_mib = int(float(preregistration["hardware"]["peak_reserved_limit_gib"]) * 1024)
    hf_peak = int(hf_cached["memory"]["peak_total_gpu_memory_mib"])
    vllm_peak = int(vllm["memory"]["peak_total_gpu_memory_mib"])
    hf_speed_passed = all(row["passed"] for row in hf_responses.values())
    external_speed_passed = all(value >= 1.2 for value in external_values)
    hf_conformance_passed = bool(hf_cached["conformance"]["threshold_passed"])
    hf_refresh = _refresh_gate(hf_cached)
    vllm_refresh = _refresh_gate(vllm)
    memory_passed = max(hf_peak, vllm_peak) <= memory_limit_mib
    teardown_passed = bool(
        hf_cached["teardown"]["backend_state"] == "closed"
        and vllm["teardown"]["backend_state"] == "closed"
        and vllm["teardown"]["port_closed"]
    )
    runtime_gate_passed = all(
        (
            hf_speed_passed,
            external_speed_passed,
            hf_conformance_passed,
            hf_refresh["passed"],
            vllm_refresh["passed"],
            memory_passed,
            teardown_passed,
        )
    )
    return {
        "schema_version": 1,
        "name": "rollout-runtime-v2-v0.11.0-candidate-rtx4080",
        "measurement_status": "completed_runtime_gate_passed"
        if runtime_gate_passed
        else "completed_runtime_gate_failed",
        "source": hf_cached["source"],
        "environment": hf_cached["environment"],
        "models": hf_cached["models"],
        "preregistration_sha256": preregistration_sha256,
        "frozen_calculator_sha256": FROZEN_CALCULATOR_SHA256,
        "artifacts": {
            "hf_reference_sha256": _sha256(BASELINE),
            "hf_cached_raw_sha256": _sha256(HF_CACHED),
            "vllm_raw_sha256": _sha256(VLLM),
        },
        "gate_interpretation": (
            "Each response-length gate passes only when every paired prompt-length, "
            "samples-per-prompt and sampling cell reaches its preregistered threshold."
        ),
        "gates": {
            "hf_cached_speedup_over_hf_reference": {
                "responses": hf_responses,
                "passed": hf_speed_passed,
            },
            "vllm_speedup_over_hf_cached": {
                "required_minimum_speedup": 1.2,
                "observed_minimum_speedup": min(external_values),
                "observed_maximum_speedup": max(external_values),
                "passed": external_speed_passed,
            },
            "hf_cached_nf4_conformance": {
                **hf_cached["conformance"],
                "passed": hf_conformance_passed,
            },
            "hf_cached_policy_refresh": hf_refresh,
            "vllm_policy_refresh": vllm_refresh,
            "peak_total_gpu_memory": {
                "limit_mib": memory_limit_mib,
                "hf_cached_peak_mib": hf_peak,
                "vllm_peak_mib": vllm_peak,
                "passed": memory_passed,
            },
            "backend_teardown": {
                "hf_cached_state": hf_cached["teardown"]["backend_state"],
                "vllm_state": vllm["teardown"]["backend_state"],
                "vllm_port_closed": vllm["teardown"]["port_closed"],
                "passed": teardown_passed,
            },
            "vllm_pg_logprob_conformance": {
                **vllm["conformance"],
                "passed": bool(vllm["conformance"]["pg_threshold_passed"]),
                "effect": "vLLM remains limited to direct GKD; PG-k1 fails closed",
            },
        },
        "backend_selection": {
            "local_default": "hf_cached",
            "external_direct_gkd": "vllm",
            "hf_cached_backend_version": hf_cached["backend"]["version"],
            "vllm_backend_version": vllm["backend"]["version"],
            "vllm_pg_k1_supported": False,
        },
        "release_progress": {
            "rollout_runtime_v2_gate_passed": runtime_gate_passed,
            "v0_11_0_published": False,
            "next_action": "run the exact-wheel full GPU qualification and release dry-run",
        },
        "cells": cells,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rendered = _bytes(build_result())
    if args.check:
        if not OUTPUT.is_file() or OUTPUT.read_bytes() != rendered:
            raise SystemExit(f"generated artifact differs: {OUTPUT}")
    else:
        OUTPUT.write_bytes(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
