from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from miniverl.cli import app

runner = CliRunner()


def _external_profile(tmp_path: Path) -> tuple[Path, Path]:
    pytest.importorskip("pyarrow")
    data = tmp_path / "prompts.parquet"
    sampled = runner.invoke(
        app,
        ["data", "sample", "--format", "verl-parquet", "--out", str(data)],
    )
    assert sampled.exit_code == 0, sampled.output

    from miniverl.bridge.opd_v08 import load_verl_opd_v08_source

    payload = load_verl_opd_v08_source("builtin:qwen3-0.6b-1.7b-opd").source.model_dump(
        mode="python"
    )
    payload["data"]["train_files"] = [str(data)]
    profile = tmp_path / "opd.yaml"
    profile.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return profile, data


def _write_plan(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, object]]:
    profile, data = _external_profile(tmp_path)
    plan = tmp_path / "plan.json"
    result = runner.invoke(
        app,
        [
            "plan",
            "--config",
            str(profile),
            "--accept-local-reinterpretations",
            "--out",
            str(plan),
            "--offline",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    return plan, profile, data, json.loads(plan.read_text(encoding="utf-8"))


def test_plan_out_binds_source_data_models_and_exact_native_config(tmp_path: Path) -> None:
    plan, profile, data, payload = _write_plan(tmp_path)

    assert payload["schema_version"] == 1
    assert payload["artifact_type"] == "miniverl_immutable_opd_plan"
    assert len(payload["plan_digest"]) == 64
    assert payload["source_config"]["path"] == str(profile.resolve())
    assert len(payload["source_config"]["sha256"]) == 64
    assert payload["data"]["manifest"]["rows"]["train"] == 4
    assert payload["data"]["files"][0]["path"] == str(data.resolve())
    assert payload["models"]["student"]["revision_kind"] == "immutable_commit"
    assert payload["models"]["teacher"]["revision_kind"] == "immutable_commit"
    assert payload["tokenizers"]["status"] == "declared_not_loaded"
    assert payload["compatibility_acceptance"]["accepted"] is True
    assert (
        payload["resolved_native_config"]["run"]["execution_plan_digest"] == payload["plan_digest"]
    )
    assert plan.read_bytes().endswith(b"\n")


def test_plan_bytes_are_deterministic_for_unchanged_inputs(tmp_path: Path) -> None:
    first, profile, _, _ = _write_plan(tmp_path)
    second = tmp_path / "second-plan.json"
    result = runner.invoke(
        app,
        [
            "plan",
            "--config",
            str(profile),
            "--accept-local-reinterpretations",
            "--out",
            str(second),
            "--offline",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert first.read_bytes() == second.read_bytes()


def test_run_from_plan_does_not_recompile_and_rejects_plan_tampering(tmp_path: Path) -> None:
    plan, _, _, payload = _write_plan(tmp_path)
    accepted = runner.invoke(
        app,
        ["run", "--plan", str(plan), "--dry-run", "--offline", "--json"],
    )
    assert accepted.exit_code == 0, accepted.output
    dry = json.loads(accepted.stdout)
    assert dry["execution_plan_digest"] == payload["plan_digest"]
    assert dry["resolved_native_config"] == payload["resolved_native_config"]

    tampered = dict(payload)
    tampered["system_plan"] = {**tampered["system_plan"], "executable": False}
    plan.write_text(json.dumps(tampered), encoding="utf-8")
    refused = runner.invoke(app, ["run", "--plan", str(plan), "--dry-run", "--offline"])
    assert refused.exit_code == 1
    assert "plan digest" in refused.output.lower()


@pytest.mark.parametrize("target", ["source", "data"])
def test_run_from_plan_rejects_source_or_data_drift(tmp_path: Path, target: str) -> None:
    plan, profile, data, _ = _write_plan(tmp_path)
    if target == "source":
        profile.write_bytes(profile.read_bytes() + b"\n# changed\n")
    else:
        data.write_bytes(data.read_bytes() + b"changed")

    refused = runner.invoke(app, ["run", "--plan", str(plan), "--dry-run", "--offline"])
    assert refused.exit_code == 1
    assert target in refused.output.lower()
    assert "changed" in refused.output.lower()


def test_run_plan_is_mutually_exclusive_with_config_and_overrides(tmp_path: Path) -> None:
    plan, profile, _, _ = _write_plan(tmp_path)
    refused = runner.invoke(
        app,
        ["run", "--plan", str(plan), "--config", str(profile), "--dry-run"],
    )
    assert refused.exit_code == 1
    assert "mutually exclusive" in refused.output.lower()

    refused = runner.invoke(
        app,
        ["run", "--plan", str(plan), "--set", "trainer.total_training_steps=3", "--dry-run"],
    )
    assert refused.exit_code == 1
    assert "override" in refused.output.lower()


def test_external_plan_requires_acceptance_before_publication(tmp_path: Path) -> None:
    profile, _ = _external_profile(tmp_path)
    target = tmp_path / "refused.json"
    refused = runner.invoke(
        app,
        ["plan", "--config", str(profile), "--out", str(target), "--offline"],
    )
    assert refused.exit_code == 1
    assert "--accept-local-reinterpretations" in refused.output
    assert not target.exists()


def test_probe_cache_transport_metadata_does_not_change_plan_digest(tmp_path: Path) -> None:
    from miniverl.bridge.opd_plan import attach_hardware_probe, build_immutable_opd_plan
    from miniverl.bridge.opd_v08 import load_verl_opd_v08_source

    profile, _ = _external_profile(tmp_path)
    compiled = load_verl_opd_v08_source(profile, accept_local_reinterpretations=True)
    base = build_immutable_opd_plan(compiled, source=profile)
    measured = {
        "status": "measured",
        "identity": {"plan_digest": base.plan_digest},
        "identity_digest": "a" * 64,
        "probe_digest": "b" * 64,
        "measurements": {"parameter_updates": 0},
        "recommendations": {},
        "failed_candidates": [],
    }
    fresh = attach_hardware_probe(base, {**measured, "cache": {"reused": False, "path": "one"}})
    reused = attach_hardware_probe(base, {**measured, "cache": {"reused": True, "path": "two"}})
    assert fresh == reused
