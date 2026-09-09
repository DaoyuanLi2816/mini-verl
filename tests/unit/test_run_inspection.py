"""Run-level inspection must expose RL records and refuse corrupt evidence."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from miniverl.cli import app
from miniverl.errors import ReportError
from miniverl.reporting.data import ReportData
from miniverl.utils.runs import canonical_json, write_json, write_text


@pytest.fixture
def run(tmp_path: Path) -> Path:
    write_json(
        tmp_path / "manifest.json",
        {
            "run_id": "ppo-example",
            "mode": "rl",
            "status": "running",
            "objective": {"name": "ppo"},
            "models": {"student": {"model_id": "tiny"}},
            "profile_identity": {"profile": "verl-rl-v0.9-single-gpu-v2"},
        },
    )
    records = [
        {"phase": "ppo_critic", "critic_update_count": 1, "value_loss": 0.2},
        {
            "phase": "rl",
            "step": 1,
            "loss": -0.1,
            "verl_rl": {"algorithm": "ppo", "actor_entropy": 0.4},
        },
        {"phase": "rl_cycle", "cycle": 0, "step": 1, "seconds": 2.0},
    ]
    import json

    write_text(tmp_path / "metrics.jsonl", "\n".join(json.dumps(row) for row in records))
    return tmp_path


def test_report_includes_actor_steps_without_counting_critic_as_actor(run: Path) -> None:
    data = ReportData.from_run(run)
    assert len(data.step_metrics) == 1
    assert data.step_metrics[0]["phase"] == "rl"
    assert data.run_inspection["updates"]["critic_records"] == 1


def test_inspect_directory_human_and_json(run: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["inspect", str(run), "--json"])
    assert result.exit_code == 0, result.output
    import json

    payload = json.loads(result.output)
    assert payload["algorithm"] == "ppo"
    assert payload["updates"]["actor_records"] == 1
    assert payload["memory"]["peak_reserved_gib"] is None
    assert payload["export"]["distributed_execution_tested"] is False
    human = runner.invoke(app, ["inspect", str(run)])
    assert human.exit_code == 0, human.output
    assert "ppo" in human.output and "critic" in human.output


@pytest.mark.parametrize(
    "bad",
    [
        '{"phase":"rl","loss":NaN}',
        '{"phase":"rl","loss":1e999}',
        '{"phase":"rl","loss":0,"loss":1}',
        "[]",
        '{"phase":',
    ],
)
def test_corrupt_metric_refuses_report_and_inspection(run: Path, bad: str) -> None:
    write_text(run / "metrics.jsonl", bad)
    with pytest.raises(ReportError):
        ReportData.from_run(run)
    result = CliRunner().invoke(app, ["inspect", str(run), "--json"])
    assert result.exit_code != 0


def test_completed_run_cannot_claim_a_missing_checkpoint(run: Path) -> None:
    import json

    manifest = json.loads((run / "manifest.json").read_text())
    manifest.update(status="completed", final_checkpoint={"directory": "final", "digest": "0" * 64})
    write_text(run / "manifest.json", canonical_json(manifest))
    with pytest.raises(ReportError, match="checkpoint"):
        ReportData.from_run(run)
