from __future__ import annotations

import json

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
