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


def test_export_and_bridge_doctor_cli_round_trip(tmp_path: Path) -> None:
    from scripts.prepare_verl_bridge_smoke import prepare_smoke_run

    source = prepare_smoke_run(tmp_path / "source")
    bundle = tmp_path / "bundle"
    exported = CliRunner().invoke(
        app,
        [
            "export-verl",
            "--run",
            str(source),
            "--target-verl",
            VERL_TAG,
            "--out",
            str(bundle),
            "--json",
        ],
    )
    assert exported.exit_code == 0, exported.output
    assert json.loads(exported.stdout)["distributed_execution_status"] == "not tested"

    diagnosed = CliRunner().invoke(app, ["bridge", "doctor", str(bundle), "--json"])
    assert diagnosed.exit_code == 0, diagnosed.output
    payload = json.loads(diagnosed.stdout)
    assert payload["verdict"] == "ok"
    assert payload["config_profile"]["model_handoff_problems"] == []


def test_convert_dataset_cli_preserves_the_official_chat_schema(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    source = tmp_path / "verl.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "data_source": "calculator",
                    "prompt": [{"role": "user", "content": "2+2"}],
                    "ability": "math",
                    "reward_model": {"style": "rule", "ground_truth": "4"},
                    "extra_info": {"miniverl": {"typed_provenance": True}},
                }
            ]
        ),
        source,
    )
    local = tmp_path / "local.parquet"
    imported = CliRunner().invoke(
        app,
        ["convert-dataset", "--from", "verl-parquet", str(source), "--out", str(local)],
    )
    assert imported.exit_code == 0, imported.output
    exported = tmp_path / "exported.parquet"
    round_trip = CliRunner().invoke(
        app,
        ["convert-dataset", "--to", "verl-parquet", str(local), "--out", str(exported)],
    )
    assert round_trip.exit_code == 0, round_trip.output
    row = pq.read_table(exported).to_pylist()[0]
    assert row["prompt"] == [{"role": "user", "content": "2+2"}]
    assert row["extra_info"]["miniverl"] == {"typed_provenance": True}
