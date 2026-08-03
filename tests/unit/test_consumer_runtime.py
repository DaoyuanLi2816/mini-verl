"""Consumer-runtime preregistration and result-contract checks."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_consumer_runtime_preregistration_freezes_the_orthogonal_matrix() -> None:
    prereg = yaml.safe_load(
        (ROOT / "benchmarks/preregistration/consumer-runtime-v1.yaml").read_text(encoding="utf-8")
    )
    assert prereg["status"] == "preregistered_before_headline_measurement"
    assert prereg["matrix"]["model_ownership"] == [
        "dual_model_resident",
        "shared_backbone_resident",
    ]
    assert prereg["matrix"]["physical_trajectory_batch"] == [1, 2, 4, "auto"]
    assert prereg["claim_scope"]["forbidden"]
    assert prereg["model"]["frozen_teacher_adapter"]["revision"] == (
        "e277b92d8c1fdb76cd133f872f0ddd2c47a4ab8c"
    )
    assert prereg["preregistration_revision"] == 1.2
    assert prereg["model"]["dtype"] == "float32"


def test_consumer_runtime_result_schema_accepts_all_prespecified_statuses() -> None:
    schema = json.loads(
        (ROOT / "benchmarks/schema/consumer-runtime-result.schema.json").read_text(encoding="utf-8")
    )
    cells = [
        {"runtime": runtime, "batch_size": batch, "status": "completed"}
        for runtime in ("dual_model", "shared_backbone")
        for batch in (1, 2, 4, "auto")
    ]
    cells.append({"runtime": "dual_model_swap", "batch_size": "auto", "status": "not_run"})
    payload = {
        "schema_version": 1,
        "name": "consumer-runtime-v1",
        "measurement_status": "measured_final",
        "preregistration_sha256": "a" * 64,
        "code_commit": "b" * 40,
        "frozen_calculator_sha256": (
            "53fc1d4d5b7adee09618d77ad62d4086ba56b78569832d6fc7c3bcd5c2695bbc"
        ),
        "environment": {},
        "workload_invariants": {
            "trajectory_digests_identical": True,
            "teacher_target_digests_identical": True,
            "trajectory_digest": "c" * 64,
            "teacher_target_digest": "d" * 64,
        },
        "equivalence_gate": {
            "passed": True,
            "comparisons_expected": 12,
            "comparisons_observed": 12,
            "tolerances": {},
            "max_observed": {},
            "failures": [],
        },
        "cells": cells,
        "larger_model_diagnostic": [
            {"size": "4B", "status": "not_run"},
            {"size": "7B", "status": "not_run"},
        ],
    }
    jsonschema.validate(payload, schema)
