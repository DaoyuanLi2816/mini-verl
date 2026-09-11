"""Native composition uses pinned data, never upstream launchers or constructors."""

from __future__ import annotations

import hashlib
import json

import pytest
import yaml

from miniverl.errors import ConfigError


def test_packaged_tree_composes_deterministically_with_upstream_overrides():
    from miniverl.bridge.hydra import compose_verl

    overrides = [
        "algorithm.adv_estimator=grpo",
        "actor_rollout_ref.rollout.n=8",
        "actor_rollout_ref.actor.optim.lr=1e-6",
        "trainer.total_epochs=1",
        "data.train_files=[/tmp/train.parquet,/tmp/other.parquet]",
        "data.shuffle=false",
        "trainer.total_training_steps=null",
        '+local_note={label:"hello world",enabled:true}',
        "++local_note.count=2",
        "~local_note.enabled",
        "actor_rollout_ref.rollout.n=4",
    ]
    first = compose_verl(config_name="ppo_trainer", overrides=overrides)
    second = compose_verl(config_name="ppo_trainer", overrides=overrides)
    assert first == second
    value = yaml.safe_load(first.resolved_yaml)
    assert value["actor_rollout_ref"]["rollout"]["n"] == 4
    assert value["actor_rollout_ref"]["actor"]["optim"]["lr"] == 1e-6
    assert value["local_note"] == {"label": "hello world", "count": 2}
    assert value["data"]["shuffle"] is False
    assert value["trainer"]["total_training_steps"] is None
    assert first.provenance["overrides"] == overrides
    assert first.provenance["resolved_sha256"] == hashlib.sha256(first.resolved_yaml).hexdigest()
    assert first.provenance["composition_status"] == "resolved"
    assert first.provenance["config_defaults"]


@pytest.mark.parametrize(
    "override",
    [
        "actor_rollout_ref.rollout.n=range(1,4)",
        "actor_rollout_ref.rollout.n=1,2",
        "hydra.searchpath=[pkg://evil]",
        "++x=${oc.env:SECRET_TOKEN}",
        "++x=${evil:command}",
        "++x=${does_not_exist}",
        "++x=.nan",
        "this is not an override",
    ],
)
def test_invalid_or_unsafe_composition_never_partially_applies(override):
    from miniverl.bridge.hydra import compose_verl

    with pytest.raises(ConfigError):
        compose_verl(config_name="ppo_trainer", overrides=[override])


def test_config_tree_version_mismatch_fails_before_composition(tmp_path):
    from miniverl.bridge.hydra import compose_verl, packaged_tree

    for name, content in packaged_tree()["files"].items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content["text"], encoding="utf-8")
    (tmp_path / "ppo_trainer.yaml").write_text("defaults: []\nalgorithm: {}\n")
    with pytest.raises(ConfigError, match=r"pinned|version"):
        compose_verl(config_path=tmp_path, config_name="ppo_trainer", overrides=[])


def test_custom_config_defaults_are_confined_and_not_instantiated(tmp_path):
    from miniverl.bridge.hydra import compose_verl, packaged_tree

    for name, content in packaged_tree()["files"].items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content["text"], encoding="utf-8")
    custom = tmp_path / "local.yaml"
    custom.write_text("defaults: [ppo_trainer, _self_]\ntrainer:\n  total_epochs: 3\n")
    result = compose_verl(config_path=tmp_path, config_name="local", overrides=[])
    assert yaml.safe_load(result.resolved_yaml)["trainer"]["total_epochs"] == 3
    assert result.provenance["config_path"] == str(tmp_path)
    assert "local.yaml" in result.provenance["input_file_sha256"]
    exact_path = tmp_path.as_posix() + "/."
    assert (
        compose_verl(config_path=exact_path, config_name="local", overrides=[]).provenance[
            "config_path"
        ]
        == exact_path
    )
    custom.write_text("defaults: [../outside]\n")
    with pytest.raises(ConfigError):
        compose_verl(config_path=tmp_path, config_name="local", overrides=[])
    custom.write_text("!!python/object/apply:os.system [echo BAD]\n")
    with pytest.raises(ConfigError):
        compose_verl(config_path=tmp_path, config_name="local", overrides=[])


def test_cli_native_hydra_has_three_status_axes():
    from typer.testing import CliRunner

    from miniverl.cli import app

    result = CliRunner().invoke(
        app,
        [
            "run",
            "--verl-config-name",
            "ppo_trainer",
            "algorithm.adv_estimator=grpo",
            "actor_rollout_ref.rollout.n=4",
            "--dry-run",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["composition_status"] == "resolved"
    assert report["semantic_status"] == "accepted"
    assert report["execution_status"] == "requires_bindings"


@pytest.mark.parametrize(
    "override",
    [
        "++api_key=ordinary-secret",
        "++nested.access_token=ordinary-secret",
        "++balance_dp_token=ordinary-secret",
    ],
)
def test_credentials_are_rejected_before_worker_invocation(monkeypatch, override):
    from miniverl.bridge import hydra

    def forbidden(*args, **kwargs):
        pytest.fail("unsafe config reached composition subprocess")

    monkeypatch.setattr(hydra.subprocess, "run", forbidden)
    with pytest.raises(ConfigError, match="credentials") as failure:
        hydra.compose_verl(config_name="ppo_trainer", overrides=[override])
    assert "ordinary-secret" not in str(failure.value)


def test_duplicate_config_keys_fail_closed():
    from miniverl.bridge.hydra import _yaml

    with pytest.raises(ConfigError, match="duplicate"):
        _yaml(b"x: 1\nx: 2\n", name="local.yaml")


def test_wrong_hydra_runtime_version_is_actionable(monkeypatch):
    import importlib.metadata

    from miniverl.bridge.hydra import compose_verl

    monkeypatch.setattr(importlib.metadata, "version", lambda name: "1.4.0")
    with pytest.raises(ConfigError, match=r"1\.3\.2"):
        compose_verl(config_name="ppo_trainer", overrides=[])


@pytest.mark.parametrize("algorithm", ["ppo", "grpo"])
def test_qualification_launch_is_semantically_accepted(tmp_path, algorithm):
    from miniverl.bridge.direct_v5 import compile_direct
    from miniverl.bridge.hydra import compose_verl
    from miniverl.bridge.hydra_examples import qualification_overrides

    composed = compose_verl(config_name="ppo_trainer", overrides=qualification_overrides(algorithm))
    path = tmp_path / "resolved.yaml"
    path.write_bytes(composed.resolved_yaml)
    report = compile_direct(
        path, bindings=["reward.provider=target_length", "sampling.max_attempted_groups=64"]
    )
    assert report["semantic_status"] == "accepted", report["rejections"]
    assert {row["field"] for row in report["required_bindings"]} == {
        "data.train_files",
        "data.val_files",
    }
