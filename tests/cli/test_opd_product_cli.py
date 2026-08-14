from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from miniverl.cli import app

runner = CliRunner()


def test_plan_builtin_is_a_no_network_compiler_smoke() -> None:
    result = runner.invoke(
        app,
        [
            "plan",
            "--profile",
            "verl-opd-v0.8-single-gpu-v1",
            "--config",
            "builtin:qwen3-0.6b-1.7b-opd",
            "--offline",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["executable"] is True
    assert payload["memory"]["status"] == "estimated"


def test_run_dry_compiles_a_valid_native_recipe() -> None:
    result = runner.invoke(
        app,
        [
            "run",
            "--profile",
            "verl-opd-v0.8-single-gpu-v1",
            "--config",
            "builtin:qwen3-0.6b-1.7b-opd",
            "--dry-run",
            "--offline",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["resolved_native_config"]["source"]["kind"] == "verl_parquet"
    assert payload["resolved_native_config"]["loss"]["mode"] == "forward_kl_topk"


def test_plan_accepts_trailing_hydra_style_overrides_and_override_files(tmp_path) -> None:
    override_file = tmp_path / "plan.overrides"
    override_file.write_text(
        "distillation.distillation_loss.topk=16\n",
        encoding="utf-8",
    )
    result = runner.invoke(
        app,
        [
            "plan",
            "--profile",
            "verl-opd-v0.8-single-gpu-v1",
            "--config",
            "builtin:qwen3-0.6b-1.7b-opd",
            "--overrides-file",
            str(override_file),
            "--set",
            "distillation.distillation_loss.topk=32",
            "--json",
            "--",
            "distillation.distillation_loss.topk=64",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["loss"]["top_k"] == 64
    assert [item["source_kind"] for item in payload["overrides"]] == [
        "overrides_file",
        "set",
        "trailing",
    ]
    assert payload["reinterpretation_acceptance"]["accepted"] is True


def test_external_run_requires_explicit_high_risk_reinterpretation_acceptance() -> None:
    config = Path(__file__).resolve().parents[2] / "examples/verl-opd-v0.8-single-gpu.yaml"
    refused = runner.invoke(
        app,
        ["run", "--config", str(config), "--dry-run", "--offline"],
    )
    assert refused.exit_code == 1
    assert "--accept-local-reinterpretations" in refused.output

    accepted = runner.invoke(
        app,
        [
            "run",
            "--config",
            str(config),
            "--dry-run",
            "--offline",
            "--accept-local-reinterpretations",
            "--json",
        ],
    )
    assert accepted.exit_code == 0, accepted.output
    payload = json.loads(accepted.stdout)
    acceptance = payload["compatibility"]["reinterpretation_acceptance"]
    assert acceptance["accepted"] is True
    assert acceptance["source"] == "cli_flag"
    assert "actor_rollout_ref.rollout.name" in acceptance["required_fields"]
    assert "actor_rollout_ref.actor.ppo_mini_batch_size" not in acceptance["required_fields"]


def test_builtin_approval_is_value_bound_and_does_not_approve_semantic_drift() -> None:
    approved = runner.invoke(
        app,
        [
            "run",
            "--config",
            "builtin:qwen3-0.6b-1.7b-opd",
            "--dry-run",
            "--offline",
            "--json",
        ],
    )
    assert approved.exit_code == 0, approved.output
    assert (
        json.loads(approved.stdout)["compatibility"]["reinterpretation_acceptance"]["source"]
        == "packaged_approval_manifest"
    )

    drifted = runner.invoke(
        app,
        [
            "run",
            "--config",
            "builtin:qwen3-0.6b-1.7b-opd",
            "--set",
            "actor_rollout_ref.rollout.name=sglang",
            "--dry-run",
            "--offline",
        ],
    )
    assert drifted.exit_code == 1
    assert "--accept-local-reinterpretations" in drifted.output


def test_data_sample_writes_portable_message_rows(tmp_path) -> None:
    pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    target = tmp_path / "sample.parquet"
    result = runner.invoke(
        app, ["data", "sample", "--format", "verl-parquet", "--out", str(target)]
    )
    assert result.exit_code == 0, result.output
    rows = pq.read_table(target).to_pylist()
    assert len(rows) == 4
    assert rows[0]["prompt"][1]["role"] == "user"
    assert rows[0]["data_source"] == "miniverl_quickstart"
    assert "reward_model" not in rows[0]
