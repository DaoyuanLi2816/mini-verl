"""Unresolved Hydra/OmegaConf interpolation must never reach a runnable recipe."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from miniverl.errors import ConfigError

PROFILE = "single-gpu-online-distillation-v1"


def _source(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
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
    for dotted, value in overrides.items():
        cursor: Any = payload
        parts = dotted.split("__")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return payload


def _write(tmp_path: Path, payload: dict[str, Any]) -> Path:
    source = tmp_path / "verl.yaml"
    source.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return source


def _resolved_choices() -> dict[str, str]:
    return {
        "environment": "calculator",
        "teacher_model": "Qwen/Qwen3-1.7B",
        "loss_profile": "topk-tail-reverse-kl",
        "schedule_mapping": "epochs-as-cycles",
    }


# --------------------------------------------------------------- source fields


@pytest.mark.parametrize(
    ("dotted", "value"),
    [
        ("actor_rollout_ref__model__path", "${MODEL_PATH}"),
        (
            "actor_rollout_ref__model__path",
            "${oc.select:model.path,Qwen/Qwen3-0.6B}",
        ),
        ("actor_rollout_ref__actor__optim__lr", "${optimizer.lr}"),
        ("trainer__project_name", "${oc.env:PROJECT}"),
        ("trainer__experiment_name", "prefix-${env:EXPERIMENT}"),
        ("data__max_response_length", "${LEN}"),
        ("data__seed", "${SEED}"),
    ],
)
def test_unresolved_interpolation_in_mapped_fields_is_rejected(
    tmp_path: Path, dotted: str, value: str
) -> None:
    """Any exact/derived/confirmed field carrying ``${...}`` must fail closed."""
    from miniverl.bridge.config import import_verl_config
    from miniverl.bridge.contract import VERL_TAG

    out = tmp_path / "recipes" / "imported.yaml"
    with pytest.raises(ConfigError, match="interpolation"):
        import_verl_config(
            _write(tmp_path, _source(**{dotted: value})),
            profile=PROFILE,
            target_verl=VERL_TAG,
            out=out,
            **_resolved_choices(),
        )
    assert not out.exists()
    assert not (out.parent / "imported.template.yaml").exists()


def test_unresolved_interpolation_is_rejected_before_the_template_branch(
    tmp_path: Path,
) -> None:
    """Missing user choices must not downgrade an interpolation defect to a template."""
    from miniverl.bridge.config import import_verl_config
    from miniverl.bridge.contract import VERL_TAG

    out = tmp_path / "recipes" / "imported.yaml"
    with pytest.raises(ConfigError, match="interpolation"):
        import_verl_config(
            _write(tmp_path, _source(actor_rollout_ref__model__path="${MODEL_PATH}")),
            profile=PROFILE,
            target_verl=VERL_TAG,
            out=out,
        )
    assert not out.exists()
    assert not (out.parent / "imported.template.yaml").exists()


def test_nested_unsupported_interpolation_is_rejected(tmp_path: Path) -> None:
    from miniverl.bridge.config import import_verl_config
    from miniverl.bridge.contract import VERL_TAG

    payload = _source()
    payload["nested"] = {"value": {"key": "${env:SECRET}"}}
    out = tmp_path / "imported.yaml"
    with pytest.raises(ConfigError):
        import_verl_config(
            _write(tmp_path, payload),
            profile=PROFILE,
            target_verl=VERL_TAG,
            out=out,
            **_resolved_choices(),
        )
    assert not out.exists()


def test_secret_bearing_interpolation_is_never_resolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The importer must not read the environment to satisfy ``${env:...}``."""
    from miniverl.bridge.config import import_verl_config
    from miniverl.bridge.contract import VERL_TAG

    monkeypatch.setenv("MODEL_PATH", "Qwen/Qwen3-0.6B")
    out = tmp_path / "imported.yaml"
    with pytest.raises(ConfigError, match="interpolation"):
        import_verl_config(
            _write(tmp_path, _source(actor_rollout_ref__model__path="${oc.env:MODEL_PATH}")),
            profile=PROFILE,
            target_verl=VERL_TAG,
            out=out,
            **_resolved_choices(),
        )
    assert not out.exists()


# ------------------------------------------------------- informational labelling


def test_informational_only_interpolation_is_labelled_and_not_executed(
    tmp_path: Path,
) -> None:
    """Informational paths may stay unresolved but must be labelled and excluded."""
    from miniverl.bridge.config import import_verl_config
    from miniverl.bridge.contract import VERL_TAG

    payload = _source()
    payload["data"]["train_files"] = ["${DATA_ROOT}/train.parquet"]
    payload["data"]["val_files"] = ["${DATA_ROOT}/val.parquet"]
    out = tmp_path / "recipes" / "imported.yaml"
    report = import_verl_config(
        _write(tmp_path, payload),
        profile=PROFILE,
        target_verl=VERL_TAG,
        out=out,
        **_resolved_choices(),
    )

    assert report["status"] == "accepted"
    classification = report["field_classification"]
    assert classification["data.train_files"]["resolution_status"] == (
        "unresolved_informational_only"
    )
    assert classification["data.val_files"]["resolution_status"] == (
        "unresolved_informational_only"
    )
    assert classification["data.seed"]["resolution_status"] == "resolved"
    assert "${" not in out.read_text(encoding="utf-8")


def test_accepted_recipe_never_contains_an_interpolation_token(tmp_path: Path) -> None:
    from miniverl.bridge.config import import_verl_config
    from miniverl.bridge.contract import VERL_TAG

    out = tmp_path / "recipes" / "imported.yaml"
    report = import_verl_config(
        _write(tmp_path, _source()),
        profile=PROFILE,
        target_verl=VERL_TAG,
        out=out,
        **_resolved_choices(),
    )
    assert report["status"] == "accepted"
    assert report["interpolation_audit"]["runnable_output_clean"] is True
    assert "${" not in out.read_text(encoding="utf-8")


# ------------------------------------------------------------------ CLI inputs


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("environment", "${ENVIRONMENT}"),
        ("teacher_model", "${TEACHER}"),
        ("teacher_adapter", "${ADAPTER_ROOT}/adapter"),
        ("loss_profile", "${LOSS}"),
        ("schedule_mapping", "${SCHEDULE}"),
    ],
)
def test_unresolved_interpolation_in_cli_choices_is_rejected(
    tmp_path: Path, option: str, value: str
) -> None:
    from miniverl.bridge.config import import_verl_config
    from miniverl.bridge.contract import VERL_TAG

    choices = _resolved_choices()
    choices[option] = value
    out = tmp_path / "imported.yaml"
    with pytest.raises(ConfigError, match="interpolation"):
        import_verl_config(
            _write(tmp_path, _source()),
            profile=PROFILE,
            target_verl=VERL_TAG,
            out=out,
            **choices,
        )
    assert not out.exists()


def test_rejection_report_records_the_interpolation_finding(tmp_path: Path) -> None:
    from miniverl.bridge.config import import_verl_config
    from miniverl.bridge.contract import VERL_TAG

    out = tmp_path / "recipes" / "imported.yaml"
    with pytest.raises(ConfigError):
        import_verl_config(
            _write(tmp_path, _source(actor_rollout_ref__model__path="${MODEL_PATH}")),
            profile=PROFILE,
            target_verl=VERL_TAG,
            out=out,
            **_resolved_choices(),
        )
    report = json.loads((out.parent / "imported.import-report.json").read_text(encoding="utf-8"))
    assert report["status"] == "rejected"
    findings = report["interpolation_audit"]["blocking"]
    assert [item["location"] for item in findings] == ["actor_rollout_ref.model.path"]
    assert findings[0]["token"] == "${MODEL_PATH}"
    assert report["generated_recipe_validated"] is False


# ------------------------------------------------------------- numeric contract


@pytest.mark.parametrize("learning_rate", ["1e-5", "1.0e-5", " 2.5E-4 "])
def test_finite_scientific_notation_strings_remain_accepted(
    tmp_path: Path, learning_rate: str
) -> None:
    from miniverl.bridge.config import import_verl_config
    from miniverl.bridge.contract import VERL_TAG

    out = tmp_path / "imported.yaml"
    report = import_verl_config(
        _write(tmp_path, _source(actor_rollout_ref__actor__optim__lr=learning_rate)),
        profile=PROFILE,
        target_verl=VERL_TAG,
        out=out,
        **_resolved_choices(),
    )
    assert report["status"] == "accepted"
    assert yaml.safe_load(out.read_text(encoding="utf-8"))["train"]["learning_rate"] == float(
        learning_rate
    )


@pytest.mark.parametrize("learning_rate", ["nan", ".inf", "-.inf", "Infinity", "-Infinity"])
def test_non_finite_numeric_values_are_rejected(tmp_path: Path, learning_rate: str) -> None:
    from miniverl.bridge.config import import_verl_config
    from miniverl.bridge.contract import VERL_TAG

    with pytest.raises(ConfigError, match="finite"):
        import_verl_config(
            _write(tmp_path, _source(actor_rollout_ref__actor__optim__lr=learning_rate)),
            profile=PROFILE,
            target_verl=VERL_TAG,
            out=tmp_path / "imported.yaml",
            **_resolved_choices(),
        )


# ------------------------------------------------------------ audit primitives


def test_audit_walks_strings_lists_tuples_and_mappings() -> None:
    from miniverl.bridge.interpolation import audit_interpolation

    payload = {
        "a": "plain",
        "b": ["ok", "${TOKEN}"],
        "c": ("fine", {"d": "${OTHER}"}),
        "e": {"f": {"g": 3}},
    }
    findings = audit_interpolation(payload, label="payload")
    assert [item["location"] for item in findings] == ["payload.b[1]", "payload.c[1].d"]
    assert [item["token"] for item in findings] == ["${TOKEN}", "${OTHER}"]


def test_audit_is_conservative_about_partial_tokens() -> None:
    from miniverl.bridge.interpolation import contains_interpolation

    assert contains_interpolation("${a")
    assert contains_interpolation("${nested:${inner}}")
    assert contains_interpolation("literal ${x} suffix")
    assert not contains_interpolation("$notinterpolation")
    assert not contains_interpolation("{braces}")
    assert not contains_interpolation("cost is $5")
