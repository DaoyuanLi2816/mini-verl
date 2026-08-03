from __future__ import annotations

import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from miniverl.bridge.contract import BRIDGE_PROFILE, VERL_TAG
from miniverl.cli import app


def test_import_verl_cli_writes_recipe_and_report(tmp_path: Path) -> None:
    source = tmp_path / "verl.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "train_files": ["train.parquet"],
                    "val_files": ["val.parquet"],
                    "prompt_key": "prompt",
                    "max_prompt_length": 64,
                    "max_response_length": 32,
                    "seed": 7,
                },
                "actor_rollout_ref": {
                    "model": {
                        "path": "Qwen/Qwen3-0.6B",
                        "enable_gradient_checkpointing": True,
                    },
                    "actor": {"optim": {"lr": 1e-5}},
                },
                "trainer": {
                    "save_freq": 1,
                    "test_freq": 1,
                    "project_name": "test",
                    "experiment_name": "bridge",
                    "total_epochs": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "imported.yaml"
    result = CliRunner().invoke(
        app,
        [
            "import-verl",
            str(source),
            "--profile",
            BRIDGE_PROFILE,
            "--target-verl",
            VERL_TAG,
            "--out",
            str(out),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["profile"] == BRIDGE_PROFILE
    assert out.is_file()
    assert (tmp_path / "import-report.json").is_file()


def test_benchmark_export_community_exact_command_needs_no_training_stack(
    tmp_path: Path,
) -> None:
    out = tmp_path / "submission.json"
    result = CliRunner().invoke(app, ["benchmark", "--export-community", str(out), "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["submission"]["measured_status"] == "not_measured"
    assert out.is_file()
