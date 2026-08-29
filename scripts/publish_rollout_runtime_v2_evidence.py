#!/usr/bin/env python3
"""Validate and aggregate the measured Rollout Runtime v2 evidence."""

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
HF_CACHED = ROOT / "benchmarks/evidence/rollout-runtime-v2/hf-cached-rtx4080-raw.json"
VLLM = ROOT / "benchmarks/evidence/rollout-runtime-v2/vllm-rtx4080-raw.json"
OUTPUT = ROOT / "benchmarks/results/rollout-runtime-v2-rtx4080.json"
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


def build_result() -> dict[str, Any]:
    preregistration = yaml.safe_load(PREREGISTRATION.read_text(encoding="utf-8"))
    baseline = _load(BASELINE)
    hf_cached = _load(HF_CACHED)
    vllm = _load(VLLM)
    expected = _expected_cells(preregistration)
    if len(expected) != 24:
        raise ValueError("the preregistered workload is no longer the frozen 24-cell matrix")
    baseline_cells = _cells(baseline, expected)
    hf_cells = _cells(hf_cached, expected)
    vllm_cells = _cells(vllm, expected)
    if hf_cached["source"] != vllm["source"]:
        raise ValueError("selected backends were not measured from the same source and wheel")
    if hf_cached["environment"] != vllm["environment"]:
        raise ValueError("selected backends were not measured in the same software environment")
    if _sha256(FROZEN_CALCULATOR) != FROZEN_CALCULATOR_SHA256:
        raise ValueError("the immutable calculator benchmark changed")

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

    hf_gate_rows: dict[str, dict[str, Any]] = {}
    for response in (256, 512):
        speedups = [
            row["hf_cached_speedup_over_hf_reference"]
            for row in cells
            if row["response_bound"] == response
        ]
        hf_gate_rows[str(response)] = {
            "required_minimum_speedup": 2.0,
            "observed_minimum_speedup": min(speedups),
            "observed_maximum_speedup": max(speedups),
            "passed": all(value >= 2.0 for value in speedups),
        }
    external_speedups = [
        row["vllm_speedup_over_hf_cached"] for row in cells if row["response_bound"] in (256, 512)
    ]
    vllm_conformance = vllm["conformance"]
    refresh = vllm["refresh_probe"]
    hf_peak = int(hf_cached["memory"]["peak_total_gpu_memory_mib"])
    vllm_peak = int(vllm["memory"]["peak_total_gpu_memory_mib"])
    memory_limit_mib = int(float(preregistration["hardware"]["peak_reserved_limit_gib"]) * 1024)
    hf_passed = all(row["passed"] for row in hf_gate_rows.values())
    external_passed = all(value >= 1.2 for value in external_speedups)
    return {
        "schema_version": 1,
        "name": "rollout-runtime-v2-rtx4080",
        "measurement_status": "completed_release_blocked",
        "source": hf_cached["source"],
        "environment": hf_cached["environment"],
        "models": hf_cached["models"],
        "preregistration_sha256": _sha256(PREREGISTRATION),
        "frozen_calculator_sha256": FROZEN_CALCULATOR_SHA256,
        "artifacts": {
            "hf_reference_sha256": _sha256(BASELINE),
            "hf_cached_raw_sha256": _sha256(HF_CACHED),
            "vllm_raw_sha256": _sha256(VLLM),
        },
        "gate_interpretation": (
            "A response-length gate passes only when every paired prompt-length, "
            "samples-per-prompt and sampling cell reaches the preregistered threshold."
        ),
        "gates": {
            "hf_cached_speedup_over_hf_reference": {
                "responses": hf_gate_rows,
                "passed": hf_passed,
            },
            "selected_external_engine_advantage_over_hf_cached": {
                "required_minimum_speedup": 1.2,
                "observed_minimum_speedup": min(external_speedups),
                "observed_maximum_speedup": max(external_speedups),
                "passed": external_passed,
            },
            "peak_total_gpu_memory": {
                "limit_mib": memory_limit_mib,
                "hf_cached_peak_mib": hf_peak,
                "vllm_peak_mib": vllm_peak,
                "passed": max(hf_peak, vllm_peak) <= memory_limit_mib,
            },
            "vllm_policy_refresh": {
                "cycles": len(refresh["cycles"]),
                "all_syncs_confirmed": refresh["all_syncs_confirmed"],
                "all_policy_identities_unique": refresh["all_policy_identities_unique"],
                "strictly_monotonic_memory_growth": refresh["strictly_monotonic_memory_growth"],
                "passed": bool(
                    refresh["all_syncs_confirmed"]
                    and refresh["all_policy_identities_unique"]
                    and not refresh["strictly_monotonic_memory_growth"]
                ),
            },
            "vllm_pg_logprob_conformance": {
                "threshold": vllm_conformance["pg_nf4_threshold"],
                "maximum_absolute_difference": vllm_conformance[
                    "sampled_logprob_max_abs_difference"
                ],
                "mean_absolute_difference": vllm_conformance["sampled_logprob_mean_abs_difference"],
                "p99_absolute_difference": vllm_conformance["sampled_logprob_p99_abs_difference"],
                "token_agreement_fraction": vllm_conformance["token_agreement_fraction"],
                "passed": vllm_conformance["pg_threshold_passed"],
                "effect": "vllm remains limited to direct GKD; PG-k1 fails closed",
            },
        },
        "backend_selection": {
            "selected": "vllm",
            "supported_scope": "managed direct GKD on the qualified WSL2 RTX 4080 stack",
            "hf_cached": {
                "status": "supported_local_backend_performance_gate_failed",
                "initial_sync_seconds": hf_cached["timing"]["initial_sync_seconds"],
                "peak_total_gpu_memory_mib": hf_peak,
            },
            "vllm": {
                "status": "selected_direct_gkd_backend",
                "version": vllm["environment"]["packages"]["vllm"],
                "initial_managed_sync_seconds": vllm["timing"]["initial_sync_seconds"],
                "isolated_startup_seconds": None,
                "startup_measurement_note": (
                    "The raw lifecycle startup field was overwritten by later no-op start "
                    "checks. Initial managed sync is valid; isolated startup is not reported."
                ),
                "peak_total_gpu_memory_mib": vllm_peak,
                "port_closed": vllm["teardown"]["port_closed"],
                "pg_k1_supported": False,
            },
            "sglang": {
                "status": "not_selected_install_blocked_before_measurement",
                "version_attempted": "0.5.18",
                "throughput": None,
                "blockers": [
                    "FlashInfer and system CUDA compatibility failed in the WSL2 spike",
                    "the Triton fallback expected /usr/local/cuda/bin/nvcc",
                ],
            },
        },
        "release_decision": {
            "v0_11_0_publishable": False,
            "blocking_gate": "hf_cached_speedup_over_hf_reference",
            "action": "publish evidence, optimize hf_cached, rerun the frozen workload",
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
