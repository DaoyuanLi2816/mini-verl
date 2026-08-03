from __future__ import annotations

import json
import struct
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml


def _safetensors_bytes() -> bytes:
    header = json.dumps(
        {"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}},
        separators=(",", ":"),
    ).encode("utf-8")
    header += b" " * ((8 - len(header) % 8) % 8)
    return struct.pack("<Q", len(header)) + header + struct.pack("<f", 0.0)


def _run(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    model = run / "model"
    data = run / "data"
    model.mkdir(parents=True)
    data.mkdir()
    (run / "manifest.json").write_text(
        json.dumps({"run_id": "bridge-source", "status": "complete"}), encoding="utf-8"
    )
    (run / "result.json").write_text(json.dumps({"strict_success": 1.0}), encoding="utf-8")
    (model / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "base_model_name_or_path": "Qwen/Qwen3-0.6B",
                "revision": "c1899de289a04d12100db370d81485cdf75e47ca",
                "target_modules": ["q_proj"],
                "r": 4,
                "lora_alpha": 8,
                "lora_dropout": 0.0,
                "bias": "none",
                "task_type": "CAUSAL_LM",
            }
        ),
        encoding="utf-8",
    )
    (model / "adapter_model.safetensors").write_bytes(_safetensors_bytes())
    (model / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "Qwen2Tokenizer"}), encoding="utf-8"
    )
    rows = [
        {
            "data_source": "calculator",
            "prompt": [{"role": "user", "content": "2+2"}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": "4"},
            "extra_info": {"split": "train"},
        }
    ]
    pq.write_table(pa.Table.from_pylist(rows), data / "train.parquet")
    pq.write_table(pa.Table.from_pylist(rows), data / "val.parquet")
    return run


def test_export_verl_emits_a_fail_closed_bundle_and_doctor_verifies_artifacts(
    tmp_path: Path,
) -> None:
    from miniverl.bridge.contract import VERL_COMMIT, VERL_TAG
    from miniverl.bridge.doctor import inspect_bridge_bundle
    from miniverl.bridge.export import export_verl_bundle

    out = tmp_path / "export"
    report = export_verl_bundle(_run(tmp_path), target_verl=VERL_TAG, out=out)

    required = {
        "model/adapter_config.json",
        "model/adapter_model.safetensors",
        "model/tokenizer_config.json",
        "model/base-model.json",
        "data/train.parquet",
        "data/val.parquet",
        "recipe/verl-overrides.yaml",
        "recipe/launch.template.sh",
        "recipe/REQUIRED_VERL.txt",
        "reward/reward_or_verifier_scaffold.py",
        "provenance/miniverl-manifest.json",
        "provenance/source-result.json",
        "provenance/compatibility-report.json",
        "provenance/SHA256SUMS",
        "README.md",
    }
    assert required.issubset(
        {path.relative_to(out).as_posix() for path in out.rglob("*") if path.is_file()}
    )
    requirement = (out / "recipe" / "REQUIRED_VERL.txt").read_text(encoding="utf-8")
    assert f"VERL_TAG={VERL_TAG}" in requirement
    assert f"VERL_COMMIT={VERL_COMMIT}" in requirement
    assert "verl-project/verl" in requirement
    assert "main" not in requirement
    assert report["compatibility_level"] == 3
    assert report["compatibility_level_name"] == "miniVERL-defined compatibility Level 3"
    assert report["artifact_bundle_complete"] is True
    assert report["upstream_config_parse_passed"] is False
    assert report["model_data_load_smoke_passed"] is False
    assert report["reward_implementation_complete"] is False
    assert report["launchable"] is False
    assert report["distributed_execution_tested"] is False
    assert report["algorithm_semantic_parity"] is False
    assert report["distributed_execution_status"] == "not tested"
    assert report["target_semantics"] == "PPO/reward scaffold"
    assert report["placeholder_defaults"]

    overrides = yaml.safe_load((out / "recipe" / "verl-overrides.yaml").read_text())
    model = overrides["actor_rollout_ref"]["model"]
    assert model["path"] == "model/base"
    assert model["lora_adapter_path"] == "model"
    assert model["lora_rank"] == 4
    assert model["lora_alpha"] == 8
    assert model["target_modules"] == ["q_proj"]
    base = json.loads((out / "model" / "base-model.json").read_text(encoding="utf-8"))
    assert base == {
        "materialized_path": "model/base",
        "model_id": "Qwen/Qwen3-0.6B",
        "revision": "c1899de289a04d12100db370d81485cdf75e47ca",
        "status": "not bundled; materialize the exact snapshot before launch",
    }
    launch = (out / "recipe" / "launch.template.sh").read_text(encoding="utf-8")
    assert "model/base/config.json" in launch
    assert "lora_adapter_path" in launch
    assert "hf download Qwen/Qwen3-0.6B --revision c1899de" in launch

    diagnosis = inspect_bridge_bundle(out)
    assert diagnosis["verdict"] == "ok"
    assert diagnosis["model_adapter_loadability"]["status"] == "ok"
    assert diagnosis["parquet_schema"]["status"] == "ok"
    assert diagnosis["reward_scaffold_importability"]["status"] == "ok"
    assert diagnosis["artifact_hashes"]["status"] == "ok"
    assert diagnosis["distributed_execution_status"] == "not tested"
    assert diagnosis["artifact_bundle_complete"] is True
    assert diagnosis["reward_implementation_complete"] is False
    assert diagnosis["launchable"] is False
    assert diagnosis["distributed_execution_tested"] is False
    assert diagnosis["algorithm_semantic_parity"] is False


def test_export_preserves_available_source_values_without_claiming_schedule_parity(
    tmp_path: Path,
) -> None:
    from miniverl.bridge.contract import VERL_TAG
    from miniverl.bridge.export import export_verl_bundle

    run = _run(tmp_path)
    (run / "config.resolved.yaml").write_text(
        yaml.safe_dump(
            {
                "rollout": {"max_total_tokens": 777, "max_new_tokens_per_turn": 96},
                "train": {
                    "cycles": 4,
                    "learning_rate": 3e-5,
                    "save_every_cycles": 2,
                    "eval_every_cycles": 3,
                },
                "environment": {"name": "json_navigation"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    out = tmp_path / "export"
    report = export_verl_bundle(run, target_verl=VERL_TAG, out=out)

    overrides = yaml.safe_load((out / "recipe" / "verl-overrides.yaml").read_text())
    assert overrides["data"]["max_response_length"] == 96
    assert overrides["actor_rollout_ref"]["actor"]["optim"]["lr"] == 3e-5
    assert report["source_run_values"]["rollout.max_total_tokens"] == 777
    assert report["source_run_values"]["train.cycles"] == 4
    assert report["source_run_values"]["environment.name"] == "json_navigation"
    placeholders = {item["field"]: item for item in report["placeholder_defaults"]}
    assert "data.max_prompt_length" in placeholders
    assert "trainer.total_epochs" in placeholders
    assert placeholders["trainer.total_epochs"]["source_run_intent"] is False


def test_current_reward_scaffold_can_never_report_ready_to_launch(tmp_path: Path) -> None:
    from miniverl.bridge.contract import VERL_TAG
    from miniverl.bridge.doctor import inspect_bridge_bundle
    from miniverl.bridge.export import export_verl_bundle

    out = tmp_path / "export"
    report = export_verl_bundle(_run(tmp_path), target_verl=VERL_TAG, out=out)
    diagnosis = inspect_bridge_bundle(out)

    assert report["reward_implementation_complete"] is False
    assert report["launchable"] is False
    assert diagnosis["reward_implementation_complete"] is False
    assert diagnosis["launchable"] is False
    assert not (out / "recipe" / "launch.sh").exists()


def test_missing_pinned_verl_does_not_relabel_a_complete_artifact_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from miniverl.bridge import doctor
    from miniverl.bridge.contract import VERL_TAG
    from miniverl.bridge.export import export_verl_bundle

    out = tmp_path / "export"
    export_verl_bundle(_run(tmp_path), target_verl=VERL_TAG, out=out)
    monkeypatch.setattr(doctor, "_installed_verl", lambda: {"status": "not installed"})

    diagnosis = doctor.inspect_bridge_bundle(out, require_verl=True)

    assert diagnosis["verdict"] == "fail"
    assert diagnosis["artifact_bundle_complete"] is True
    assert diagnosis["upstream_config_parse_passed"] is False


def test_bridge_doctor_detects_tampering(tmp_path: Path) -> None:
    from miniverl.bridge.contract import VERL_TAG
    from miniverl.bridge.doctor import inspect_bridge_bundle
    from miniverl.bridge.export import export_verl_bundle

    out = tmp_path / "export"
    export_verl_bundle(_run(tmp_path), target_verl=VERL_TAG, out=out)
    with (out / "README.md").open("a", encoding="utf-8") as fh:
        fh.write("tampered\n")
    diagnosis = inspect_bridge_bundle(out)
    assert diagnosis["verdict"] == "fail"
    assert diagnosis["artifact_hashes"]["status"] == "fail"


def test_bridge_doctor_rejects_a_rehashed_but_semantically_wrong_lora_handoff(
    tmp_path: Path,
) -> None:
    from miniverl.bridge.contract import VERL_TAG
    from miniverl.bridge.doctor import inspect_bridge_bundle
    from miniverl.bridge.export import _write_hashes, export_verl_bundle

    out = tmp_path / "export"
    export_verl_bundle(_run(tmp_path), target_verl=VERL_TAG, out=out)
    config_path = out / "recipe" / "verl-overrides.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["actor_rollout_ref"]["model"]["lora_adapter_path"] = "model/wrong"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    _write_hashes(out)

    diagnosis = inspect_bridge_bundle(out)
    assert diagnosis["artifact_hashes"]["status"] == "ok"
    assert diagnosis["config_profile"]["status"] == "fail"
    assert diagnosis["verdict"] == "fail"


def test_export_verl_requires_standard_peft_and_both_dataset_splits(tmp_path: Path) -> None:
    from miniverl.bridge.contract import VERL_TAG
    from miniverl.bridge.export import export_verl_bundle
    from miniverl.errors import ConfigError

    run = _run(tmp_path)
    (run / "data" / "val.parquet").unlink()
    with pytest.raises(ConfigError, match=r"val\.parquet"):
        export_verl_bundle(run, target_verl=VERL_TAG, out=tmp_path / "export")


def test_export_verl_rejects_an_unpinned_base_model_identity(tmp_path: Path) -> None:
    from miniverl.bridge.contract import VERL_TAG
    from miniverl.bridge.export import export_verl_bundle
    from miniverl.errors import ConfigError

    run = _run(tmp_path)
    config_path = run / "model" / "adapter_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["revision"] = None
    config_path.write_text(json.dumps(config), encoding="utf-8")
    destination = tmp_path / "export"

    with pytest.raises(ConfigError, match="immutable 40-character base revision"):
        export_verl_bundle(run, target_verl=VERL_TAG, out=destination)
    assert not destination.exists()
