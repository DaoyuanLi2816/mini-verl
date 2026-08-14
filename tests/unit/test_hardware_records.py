from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from miniverl.cli import app

_DIGEST = "a" * 64
_REVISION = "b" * 40


def _run(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    (run / "checkpoints/final").mkdir(parents=True)
    (run / "final-peft-adapter").mkdir()
    manifest = {
        "status": "completed",
        "run_id": "portable-run",
        "miniverl_version": "0.10.0.dev0",
        "execution_plan_digest": _DIGEST,
        "resolved_config_digest": "c" * 64,
        "actual_optimizer_updates": 8,
        "resumed_from": None,
        "final_checkpoint": {"digest": "d" * 64},
        "measurement_status": {"cuda_metrics": "measured"},
        "platform": "Linux-6.8",
        "python_version": "3.12.4",
        "profile_identity": {
            "profile_name": "verl-opd-v0.8-single-gpu-v1",
            "digest": "e" * 64,
            "upstream_tag": "v0.8.0",
            "upstream_commit": _REVISION,
        },
        "gpu": {
            "name": "NVIDIA GeForce RTX 4080",
            "device_count": 1,
            "total_memory_gib": 15.992,
            "driver_version": None,
            "torch_cuda_version": "13.0",
            "torch_version": "2.13.0+cu130",
        },
        "models": {
            "student": {
                "model_id": "org/student",
                "revision": _REVISION,
                "quantization": "nf4",
                "capabilities": {"dtype": "bfloat16"},
            },
            "teacher": {
                "model_id": "org/teacher",
                "revision": "c" * 40,
                "quantization": "nf4",
                "capabilities": {"dtype": "bfloat16"},
            },
        },
        "objective": {"loss_mode": "forward_kl_topk", "top_k": 32},
    }
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    config = {
        "source": {"max_prompt_length": 128, "max_response_length": 64},
        "rollout": {"prompt_batch_size": 4},
        "train": {"rollouts_per_cycle": 4, "trajectory_batch_size": 1},
        "loss": {"estimator_implementation_version": None},
    }
    (run / "config.validated.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    (run / "local-execution-plan.json").write_text(
        json.dumps({"system_plan": {"batching": {"teacher_score": 2}}}),
        encoding="utf-8",
    )
    rows = [
        {
            "phase": "opd_cycle",
            "rollout_seconds": 2.0,
            "teacher_scoring_seconds": 0.5,
            "memory": {"peak_allocated_gib": 1.1, "peak_reserved_gib": 1.3},
        },
        {
            "phase": "opd",
            "seconds": 1.0,
            "memory": {"peak_allocated_gib": 1.2, "peak_reserved_gib": 1.4},
        },
    ]
    (run / "metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    for relative in (
        "checkpoints/final/adapter.safetensors",
        "checkpoints/final/optimizer.safetensors",
        "checkpoints/final/state.json",
        "trajectories.jsonl",
        "final-peft-adapter/adapter_model.safetensors",
    ):
        (run / relative).write_bytes(relative.encode("utf-8"))
    return run


def test_record_extracts_strict_portable_measured_evidence(tmp_path: Path) -> None:
    from miniverl.evidence.hardware import build_hardware_record

    record = build_hardware_record(_run(tmp_path))
    payload = record.model_dump(mode="json")

    assert payload["record_classification"] == "community_submitted"
    assert payload["review_status"] == "unreviewed"
    assert payload["hardware"]["driver"]["status"] == "unknown"
    assert payload["batching"] == {
        "logical": 4,
        "rollout_physical": 4,
        "teacher_score_physical": 2,
        "update_trajectory_physical": 1,
    }
    assert payload["optimizer_updates"]["value"] == 8
    assert payload["measurements"]["peak_reserved_gib"]["value"] == 1.4
    assert payload["resume"] == {
        "status": "unknown",
        "outcome": "not_exercised",
        "checkpoint_digest": None,
    }
    assert len(payload["artifacts"]) == 5
    assert str(tmp_path) not in json.dumps(payload)
    assert payload["consent_to_publish"] is False


def test_validation_rejects_status_invention_and_private_paths(tmp_path: Path) -> None:
    from miniverl.evidence.hardware import (
        build_hardware_record,
        validate_hardware_record,
        write_hardware_record,
    )

    record = build_hardware_record(_run(tmp_path))
    path = write_hardware_record(record, tmp_path / "record.json")
    assert validate_hardware_record(path) == []

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["hardware"]["driver"]["value"] = "invented"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert "unknown evidence must not invent a value" in validate_hardware_record(path)[0]

    payload["hardware"]["driver"] = {
        "status": "measured",
        "value": "C:\\Users\\someone\\driver",
        "unit": None,
        "source": "manifest.json",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert validate_hardware_record(path)[0].startswith("privacy:")


def test_pg_record_reports_estimator_instead_of_internal_top_k_sentinel(tmp_path: Path) -> None:
    from miniverl.evidence.hardware import build_hardware_record

    run = _run(tmp_path)
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["objective"]["loss_mode"] = "verl_pg_k1"
    manifest["objective"]["top_k"] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    config_path = run / "config.validated.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["loss"]["estimator_implementation_version"] = "verl-v0.8-pg-k1-v1"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    target = build_hardware_record(run).target
    assert target.mode == "verl_pg_k1"
    assert target.top_k is None
    assert target.estimator == "verl-v0.8-pg-k1-v1"


def test_hardware_cli_round_trip_is_unreviewed_and_does_not_upload(tmp_path: Path) -> None:
    run = _run(tmp_path)
    out = tmp_path / "hardware-record.json"
    runner = CliRunner()

    recorded = runner.invoke(app, ["hardware", "record", "--run", str(run), "--out", str(out)])
    assert recorded.exit_code == 0, recorded.output
    assert "nothing was uploaded" in recorded.output
    validated = runner.invoke(app, ["hardware", "validate", str(out), "--json"])
    assert validated.exit_code == 0, validated.output
    payload = json.loads(validated.stdout)
    assert payload["valid"] is True
    assert payload["review_status"] == "unreviewed"


def test_maintainer_review_cannot_be_self_asserted_without_consent(tmp_path: Path) -> None:
    from miniverl.evidence.hardware import build_hardware_record

    record = build_hardware_record(_run(tmp_path))
    payload = record.model_dump(mode="json")
    payload["record_classification"] = "maintainer_measured"
    payload["review_status"] = "maintainer_validated"
    path = tmp_path / "record.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    from miniverl.evidence.hardware import validate_hardware_record

    assert "requires consent" in validate_hardware_record(path)[0]


def test_maintainer_measurement_requires_review_and_consent(tmp_path: Path) -> None:
    from pydantic import ValidationError

    from miniverl.evidence.hardware import build_hardware_record

    run = _run(tmp_path)
    with pytest.raises(ValidationError, match="requires maintainer_validated"):
        build_hardware_record(
            run,
            classification="maintainer_measured",
            consent_to_publish=True,
        )

    record = build_hardware_record(
        run,
        classification="maintainer_measured",
        review_status="maintainer_validated",
        consent_to_publish=True,
    )
    assert record.review_status == "maintainer_validated"


def test_committed_hardware_schema_is_exactly_generated() -> None:
    from scripts.publish_hardware_record_schema import render

    path = Path("docs/generated/hardware-record-v1.schema.json")
    assert path.read_text(encoding="utf-8") == render()
