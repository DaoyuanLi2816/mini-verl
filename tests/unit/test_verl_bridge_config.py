from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from miniverl.errors import ConfigError


def _source(*, learning_rate: object = 2.0e-5) -> dict[str, object]:
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
            "actor": {"optim": {"lr": learning_rate}},
        },
        "trainer": {
            "save_freq": 2,
            "test_freq": 1,
            "project_name": "bridge-smoke",
            "experiment_name": "strict-profile",
            "total_epochs": 3,
        },
    }


def _write_source(tmp_path: Path, payload: dict[str, object] | None = None) -> Path:
    source = tmp_path / "verl.yaml"
    source.write_text(yaml.safe_dump(payload or _source(), sort_keys=False), encoding="utf-8")
    return source


def test_import_verl_defaults_to_a_non_executable_needs_input_template(
    tmp_path: Path,
) -> None:
    from miniverl.bridge.config import import_verl_config
    from miniverl.bridge.contract import VERL_COMMIT, VERL_REPOSITORY, VERL_TAG
    from miniverl.config import RunConfig

    out = tmp_path / "recipes" / "imported.yaml"
    report = import_verl_config(
        _write_source(tmp_path),
        profile="single-gpu-online-distillation-v1",
        target_verl=VERL_TAG,
        out=out,
    )

    template = out.parent / "imported.template.yaml"
    assert not out.exists()
    assert template.is_file()
    with pytest.raises(ValidationError):
        RunConfig.from_yaml(template)

    on_disk = json.loads((out.parent / "imported.import-report.json").read_text(encoding="utf-8"))
    assert on_disk == report
    assert report["report_path"] == "imported.import-report.json"
    assert report["source_verl"] == {
        "repository": VERL_REPOSITORY,
        "tag": VERL_TAG,
        "commit": VERL_COMMIT,
    }
    assert report["status"] == "needs_user_input"
    assert report["generated_path"] == "imported.template.yaml"
    assert {item["field"] for item in report["required_user_input"]} == {
        "environment",
        "teacher_identity",
        "loss_profile",
        "schedule_mapping",
    }
    classifications = report["field_classification"]
    assert classifications["data.train_files"]["classification"] == "informational_only"
    assert classifications["data.val_files"]["classification"] == "informational_only"
    assert classifications["data.prompt_key"]["classification"] == "informational_only"
    assert classifications["data.max_prompt_length"]["classification"] == "derived"
    assert classifications["trainer.total_epochs"]["classification"] == (
        "requires_user_confirmation"
    )
    assert classifications["trainer.save_freq"]["classification"] == ("requires_user_confirmation")
    assert classifications["trainer.test_freq"]["classification"] == ("requires_user_confirmation")
    assert all(
        item["classification"]
        in {
            "exact",
            "derived",
            "informational_only",
            "requires_user_confirmation",
            "unsupported",
        }
        for item in classifications.values()
    )
    assert "mapped_fields" not in report
    rendered = template.read_text(encoding="utf-8")
    assert "calculator" not in rendered
    assert "teacher" not in rendered.lower() or "required" in rendered.lower()


def test_import_verl_explicit_contract_produces_a_valid_recipe(tmp_path: Path) -> None:
    from miniverl.bridge.config import import_verl_config
    from miniverl.bridge.contract import VERL_TAG
    from miniverl.config import RunConfig

    out = tmp_path / "recipes" / "imported.yaml"
    report = import_verl_config(
        _write_source(tmp_path),
        profile="single-gpu-online-distillation-v1",
        target_verl=VERL_TAG,
        out=out,
        environment="jsonnav",
        teacher_model="Qwen/Qwen3-1.7B",
        loss_profile="topk-tail-reverse-kl",
        schedule_mapping="epochs-as-cycles",
    )

    generated = yaml.safe_load(out.read_text(encoding="utf-8"))
    validated = RunConfig.from_yaml(out)
    assert generated["run"]["name"] == "bridge-smoke-strict-profile"
    assert generated["run"]["mode"] == "opd"
    assert generated["models"]["student"]["model_id"] == "Qwen/Qwen3-0.6B"
    assert generated["models"]["teacher"]["model_id"] == "Qwen/Qwen3-1.7B"
    assert generated["models"]["teacher"]["mode"] == "standard"
    assert generated["environment"]["name"] == "jsonnav"
    assert generated["rollout"]["max_new_tokens_per_turn"] == 128
    assert generated["rollout"]["max_total_tokens"] == 640
    assert generated["train"]["learning_rate"] == 2.0e-5
    assert generated["train"]["cycles"] == 3
    assert validated.environment.name == "jsonnav"
    assert report["status"] == "accepted"
    assert report["generated_path"] == "imported.yaml"
    assert report["generated_recipe_validated"] is True
    assert report["user_confirmations"]["schedule_mapping"] == "epochs-as-cycles"


def test_import_verl_accepts_finite_scientific_notation_strings(tmp_path: Path) -> None:
    from miniverl.bridge.config import import_verl_config
    from miniverl.bridge.contract import VERL_TAG

    source = tmp_path / "verl.yaml"
    source.write_text(
        yaml.safe_dump(_source(), sort_keys=False).replace("2.0e-05", "1e-5"),
        encoding="utf-8",
    )
    out = tmp_path / "imported.yaml"
    report = import_verl_config(
        source,
        profile="single-gpu-online-distillation-v1",
        target_verl=VERL_TAG,
        out=out,
        environment="calculator",
        teacher_model="Qwen/Qwen3-1.7B",
        loss_profile="topk-tail-reverse-kl",
        schedule_mapping="epochs-as-cycles",
    )
    assert report["status"] == "accepted"
    assert yaml.safe_load(out.read_text(encoding="utf-8"))["train"]["learning_rate"] == 1e-5


@pytest.mark.parametrize("learning_rate", ["nan", ".inf", "-inf", "${actor.lr}"])
def test_import_verl_rejects_non_finite_or_unresolved_numeric_values(
    tmp_path: Path, learning_rate: str
) -> None:
    from miniverl.bridge.config import import_verl_config
    from miniverl.bridge.contract import VERL_TAG

    with pytest.raises(ConfigError, match=r"finite|interpolation"):
        import_verl_config(
            _write_source(tmp_path, _source(learning_rate=learning_rate)),
            profile="single-gpu-online-distillation-v1",
            target_verl=VERL_TAG,
            out=tmp_path / "imported.yaml",
            environment="calculator",
            teacher_model="Qwen/Qwen3-1.7B",
            loss_profile="topk-tail-reverse-kl",
            schedule_mapping="epochs-as-cycles",
        )


def test_import_verl_does_not_qualify_a_same_base_teacher_without_an_adapter(
    tmp_path: Path,
) -> None:
    from miniverl.bridge.config import import_verl_config
    from miniverl.bridge.contract import VERL_TAG

    report = import_verl_config(
        _write_source(tmp_path),
        profile="single-gpu-online-distillation-v1",
        target_verl=VERL_TAG,
        out=tmp_path / "imported.yaml",
        environment="calculator",
        teacher_model="Qwen/Qwen3-0.6B",
        loss_profile="topk-tail-reverse-kl",
        schedule_mapping="epochs-as-cycles",
    )
    assert report["status"] == "needs_user_input"
    assert any(item["field"] == "teacher_identity" for item in report["required_user_input"])
    assert not (tmp_path / "imported.yaml").exists()


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

    with pytest.raises(ConfigError, match="unsupported verl field"):
        import_verl_config(
            _write_source(tmp_path, payload),
            profile="single-gpu-online-distillation-v1",
            target_verl=VERL_TAG,
            out=tmp_path / "imported.yaml",
        )
    rejection = json.loads((tmp_path / "imported.import-report.json").read_text(encoding="utf-8"))
    assert rejection["status"] == "rejected"
    assert rejection["field_classification"][".".join(path)]["classification"] == "unsupported"


def test_import_verl_rejects_a_moving_or_unverified_target(tmp_path: Path) -> None:
    from miniverl.bridge.config import import_verl_config

    with pytest.raises(ConfigError, match="pinned verl target"):
        import_verl_config(
            _write_source(tmp_path),
            profile="single-gpu-online-distillation-v1",
            target_verl="main",
            out=tmp_path / "imported.yaml",
        )
