from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from miniverl.errors import ConfigError


def _source() -> dict[str, object]:
    return {
        "data": {
            "train_files": ["train.parquet"],
            "val_files": ["val.parquet"],
            "prompt_key": "prompt",
            "max_prompt_length": 512,
            "max_response_length": 128,
            "seed": 77,
        },
        "actor_rollout_ref": {
            "model": {
                "path": "Qwen/Qwen3-0.6B",
                "enable_gradient_checkpointing": True,
            },
            "actor": {"optim": {"lr": 2.0e-5}},
        },
        "trainer": {
            "save_freq": 2,
            "test_freq": 1,
            "project_name": "bridge-smoke",
            "experiment_name": "strict-profile",
            "total_epochs": 3,
        },
    }


def test_import_verl_maps_only_the_pinned_profile_and_writes_a_report(tmp_path: Path) -> None:
    from miniverl.bridge.config import import_verl_config
    from miniverl.bridge.contract import VERL_COMMIT, VERL_REPOSITORY, VERL_TAG

    source = tmp_path / "verl.yaml"
    source.write_text(yaml.safe_dump(_source(), sort_keys=False), encoding="utf-8")
    out = tmp_path / "recipes" / "imported.yaml"

    report = import_verl_config(
        source,
        profile="single-gpu-online-distillation-v1",
        target_verl=VERL_TAG,
        out=out,
    )

    generated = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert generated["run"] == {
        "name": "bridge-smoke-strict-profile",
        "mode": "opd",
        "seed": 77,
        "output_dir": "runs",
        "deterministic": True,
        "tags": ["verl-bridge", "single-gpu-online-distillation-v1"],
    }
    assert generated["models"]["student"]["model_id"] == "Qwen/Qwen3-0.6B"
    assert generated["models"]["student"]["gradient_checkpointing"] is True
    assert generated["models"]["teacher"]["model_id"] == "Qwen/Qwen3-0.6B"
    assert generated["rollout"]["max_new_tokens_per_turn"] == 128
    assert generated["rollout"]["max_total_tokens"] == 640
    assert generated["train"]["learning_rate"] == 2.0e-5
    assert generated["train"]["cycles"] == 3
    assert generated["train"]["save_every_cycles"] == 2
    assert generated["train"]["eval_every_cycles"] == 1

    on_disk = json.loads((out.parent / "import-report.json").read_text(encoding="utf-8"))
    assert on_disk == report
    assert report["source_verl"] == {
        "repository": VERL_REPOSITORY,
        "tag": VERL_TAG,
        "commit": VERL_COMMIT,
    }
    assert report["profile"] == "single-gpu-online-distillation-v1"
    assert report["unsupported_fields"] == []
    assert report["semantic_conflicts"] == []
    assert report["source_config_sha256"]
    assert report["generated_miniverl_sha256"]
    assert report["status"] == "accepted"
    assert "data.train_files" in report["mapped_fields"]
    assert report["mapped_fields"]["data.train_files"]["disposition"] == "bridge_metadata"


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("algorithm", "adv_estimator"), "grpo"),
        (("actor_rollout_ref", "actor", "ppo_clip_ratio"), 0.2),
        (("trainer", "n_gpus_per_node"), 2),
        (("actor_rollout_ref", "rollout", "tensor_model_parallel_size"), 2),
    ],
)
def test_import_verl_fails_closed_on_algorithm_or_scale_out_fields(
    tmp_path: Path, path: tuple[str, ...], value: object
) -> None:
    from miniverl.bridge.config import import_verl_config
    from miniverl.bridge.contract import VERL_TAG

    payload = _source()
    cursor: dict[str, object] = payload
    for part in path[:-1]:
        child = cursor.setdefault(part, {})
        assert isinstance(child, dict)
        cursor = child
    cursor[path[-1]] = value
    source = tmp_path / "verl.yaml"
    source.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ConfigError, match="unsupported verl field"):
        import_verl_config(
            source,
            profile="single-gpu-online-distillation-v1",
            target_verl=VERL_TAG,
            out=tmp_path / "imported.yaml",
        )
    assert not (tmp_path / "imported.yaml").exists()
    rejection = json.loads((tmp_path / "import-report.json").read_text(encoding="utf-8"))
    assert rejection["status"] == "rejected"
    assert ".".join(path) in rejection["unsupported_fields"]


def test_import_verl_rejects_a_moving_or_unverified_target(tmp_path: Path) -> None:
    from miniverl.bridge.config import import_verl_config

    source = tmp_path / "verl.yaml"
    source.write_text(yaml.safe_dump(_source()), encoding="utf-8")
    with pytest.raises(ConfigError, match="pinned verl target"):
        import_verl_config(
            source,
            profile="single-gpu-online-distillation-v1",
            target_verl="main",
            out=tmp_path / "imported.yaml",
        )
