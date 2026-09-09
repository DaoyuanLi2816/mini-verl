"""Installed examples use the same import, planner and data commands as users."""

import json

import pytest
from typer.testing import CliRunner

from miniverl.cli import app


@pytest.mark.parametrize("algorithm,updates", [("ppo", 4), ("grpo", 2)])
def test_packaged_rl_import_and_download_free_plan(tmp_path, algorithm, updates):
    out = tmp_path / f"{algorithm}.yaml"
    runner = CliRunner()
    imported = runner.invoke(
        app,
        [
            "import-verl",
            "--example",
            algorithm,
            "--profile",
            "verl-rl-v0.9-single-gpu-v3",
            "--out",
            str(out),
            "--json",
        ],
    )
    assert imported.exit_code == 0, imported.output
    assert json.loads(imported.output)["executable"]
    assert out.with_suffix(".verl.yaml").is_file()
    provenance = json.loads(out.with_suffix(".example-provenance.json").read_text())
    assert provenance["upstream"]["path"].startswith("examples/")
    assert provenance["changes"]
    planned = runner.invoke(app, ["train", str(out), "--dry-run", "--json"])
    assert planned.exit_code == 0, planned.output
    plan = json.loads(planned.output)
    assert plan["planned_optimizer_steps"] == updates
    assert plan["planned_critic_updates"] == (4 if algorithm == "ppo" else 0)
    assert plan["planned_rollouts"] == 8


def test_sample_length_reward_metadata(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    out = tmp_path / "data.parquet"
    result = CliRunner().invoke(
        app, ["data", "sample", "--reward-profile", "target-length", "--out", str(out)]
    )
    assert result.exit_code == 0, result.output
    rows = pq.read_table(out).to_pylist()
    assert [row["reward_model"]["characters"] for row in rows] == [12, 24, 48, 72]
    assert all(row["reward_model"]["style"] == "target_length" for row in rows)
