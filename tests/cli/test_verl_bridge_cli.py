from __future__ import annotations

import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from miniverl.bridge.contract import BRIDGE_PROFILE, VERL_TAG
from miniverl.bridge.opd_v08 import VERL_OPD_V08_PROFILE
from miniverl.cli import app


def test_import_verl_cli_defaults_to_a_needs_input_template(tmp_path: Path) -> None:
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
    assert payload["status"] == "needs_user_input"
    assert not out.exists()
    # The output family is stem-specific; nothing shares a name across stems.
    assert (tmp_path / "imported.template.yaml").is_file()
    assert (tmp_path / "imported.import-report.json").is_file()
    assert not (tmp_path / "import-report.json").exists()
    assert payload["report"] == str(tmp_path / "imported.import-report.json")


def test_import_verl_cli_refuses_a_collision_without_overwrite(tmp_path: Path) -> None:
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
                    "model": {"path": "Qwen/Qwen3-0.6B"},
                    "actor": {"optim": {"lr": "1e-5"}},
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
    out = tmp_path / "recipes" / "foo.yaml"
    command = [
        "import-verl",
        str(source),
        "--profile",
        BRIDGE_PROFILE,
        "--target-verl",
        VERL_TAG,
        "--out",
        str(out),
        "--environment",
        "calculator",
        "--teacher-model",
        "Qwen/Qwen3-1.7B",
        "--loss-profile",
        "topk-tail-reverse-kl",
        "--schedule-mapping",
        "epochs-as-cycles",
    ]
    runner = CliRunner()
    assert runner.invoke(app, command).exit_code == 0
    before = out.read_bytes()

    collided = runner.invoke(app, command)
    assert collided.exit_code != 0
    assert "--overwrite" in collided.output
    assert out.read_bytes() == before

    replaced = runner.invoke(app, [*command, "--overwrite"])
    assert replaced.exit_code == 0, replaced.output
    assert out.is_file()


def test_import_verl_cli_rejects_an_unexpanded_shell_variable(tmp_path: Path) -> None:
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
                    "model": {"path": "Qwen/Qwen3-0.6B"},
                    "actor": {"optim": {"lr": "1e-5"}},
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
            "--environment",
            "${ENVIRONMENT}",
            "--teacher-model",
            "Qwen/Qwen3-1.7B",
            "--loss-profile",
            "topk-tail-reverse-kl",
            "--schedule-mapping",
            "epochs-as-cycles",
        ],
    )
    assert result.exit_code != 0
    assert "interpolation" in result.output
    assert not out.exists()
    assert not (tmp_path / "imported.import-report.json").exists()


def test_import_verl_cli_explicit_contract_writes_a_valid_recipe(tmp_path: Path) -> None:
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
                    "model": {"path": "Qwen/Qwen3-0.6B"},
                    "actor": {"optim": {"lr": "1e-5"}},
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
            "--environment",
            "calculator",
            "--teacher-model",
            "Qwen/Qwen3-1.7B",
            "--loss-profile",
            "topk-tail-reverse-kl",
            "--schedule-mapping",
            "epochs-as-cycles",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["status"] == "accepted"
    assert out.is_file()


def test_import_verl_v2_writes_a_round_trippable_prompt_opd_profile(tmp_path: Path) -> None:
    from miniverl.bridge.opd_v08 import load_verl_opd_v08

    source = Path("examples/verl-opd-v0.8-single-gpu.yaml")
    out = tmp_path / "local-opd.yaml"
    result = CliRunner().invoke(
        app,
        [
            "import-verl",
            "--config",
            str(source),
            "--profile",
            VERL_OPD_V08_PROFILE,
            "--set",
            'data.train_files=["train.parquet"]',
            "--out",
            str(out),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["status"] == "accepted"
    assert payload["generated_profile_validated"] is True
    assert payload["environment_required"] is False
    report = json.loads((tmp_path / "local-opd.import-report.json").read_text(encoding="utf-8"))
    assert report["compiled_digest"] == payload["compiled_digest"]
    round_trip = load_verl_opd_v08(out)
    assert round_trip.executable is True
    assert round_trip.source.data.train_files == ["train.parquet"]


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
    exported_payload = json.loads(exported.stdout)
    assert exported_payload["distributed_execution_status"] == "not tested"
    assert exported_payload["launchable"] is False

    diagnosed = CliRunner().invoke(app, ["bridge", "doctor", str(bundle), "--json"])
    assert diagnosed.exit_code == 0, diagnosed.output
    payload = json.loads(diagnosed.stdout)
    assert payload["verdict"] == "ok"
    assert payload["config_profile"]["model_handoff_problems"] == []
    assert payload["launchable"] is False


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
