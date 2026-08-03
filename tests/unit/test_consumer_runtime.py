"""Consumer-runtime preregistration and result-contract checks."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import jsonschema
import yaml

ROOT = Path(__file__).resolve().parents[2]


def _publisher() -> ModuleType:
    path = ROOT / "scripts/publish_consumer_runtime_artifacts.py"
    spec = importlib.util.spec_from_file_location("publish_consumer_runtime_artifacts", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_measured_consumer_runtime_artifacts_are_exact_and_data_bound() -> None:
    expected = {
        "benchmarks/results/consumer-runtime-v1.json": (
            "a302da31af99f1d29f1efd4e6b3dbeb6ea4ac956bba102ca8a1bee8dff0319eb"
        ),
        "benchmarks/results/consumer-runtime-v1-profiler.json": (
            "66111cd7fc876cf1befea3297a1a51bcd99252c0bf8989c029381e1dc155a98b"
        ),
        "docs/consumer-runtime-v1-pareto.svg": (
            "98645a668a7832423d28b621262292619615917f037adf7219ff1bf071fb2fea"
        ),
    }
    for relative, digest in expected.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest

    result_path = ROOT / "benchmarks/results/consumer-runtime-v1.json"
    profiler_path = ROOT / "benchmarks/results/consumer-runtime-v1-profiler.json"
    result_schema = json.loads(
        (ROOT / "benchmarks/schema/consumer-runtime-result.schema.json").read_text(encoding="utf-8")
    )
    profiler_schema = json.loads(
        (ROOT / "benchmarks/schema/consumer-runtime-profiler.schema.json").read_text(
            encoding="utf-8"
        )
    )
    jsonschema.validate(json.loads(result_path.read_text(encoding="utf-8")), result_schema)
    jsonschema.validate(json.loads(profiler_path.read_text(encoding="utf-8")), profiler_schema)
    payload = _publisher()._load_result(result_path)
    assert payload["code_commit"] == "e44584b04837a05b0dd834c7948666d843908486"
    assert payload["equivalence_gate"]["comparisons_observed"] == 12
    assert payload["equivalence_gate"]["max_observed"] == {
        "gradient_max_absolute_difference": 7.227063179016113e-06,
        "gradient_max_relative_to_reference_max": 2.802595029641054e-05,
        "loss_absolute_difference": 1.2479722499847412e-06,
        "updated_logit_max_absolute_difference": 0.00012993812561035156,
        "updated_logit_max_relative_to_reference_max": 3.953026727392894e-06,
    }
    rendered = _publisher().render_pareto(
        payload, hashlib.sha256(result_path.read_bytes()).hexdigest()
    )
    assert (ROOT / "docs/consumer-runtime-v1-pareto.svg").read_text(encoding="utf-8") == rendered
