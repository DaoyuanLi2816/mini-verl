from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from miniverl.cli import app

runner = CliRunner()
PROFILE = "verl-opd-v0.8-single-gpu-v1"


def test_profiles_list_show_and_schema_are_copyable_and_machine_readable() -> None:
    listed = runner.invoke(app, ["profiles", "list", "--json"])
    assert listed.exit_code == 0, listed.output
    assert json.loads(listed.stdout)[0]["name"] == PROFILE

    shown = runner.invoke(app, ["profiles", "show", PROFILE, "--json"])
    assert shown.exit_code == 0, shown.output
    payload = json.loads(shown.stdout)
    assert payload["identity"]["profile_name"] == PROFILE
    assert "data:" in payload["minimal_yaml"]
    assert payload["override_invocation"].startswith("miniverl plan")

    schema = runner.invoke(app, ["profiles", "schema", PROFILE, "--json"])
    assert schema.exit_code == 0, schema.output
    assert json.loads(schema.stdout)["title"] == "VerlOPDV08Profile"


def test_compat_explain_and_check_distinguish_field_statuses() -> None:
    explained = runner.invoke(
        app,
        [
            "compat",
            "explain",
            "--profile",
            PROFILE,
            "actor_rollout_ref.actor.ppo_mini_batch_size",
            "--json",
        ],
    )
    assert explained.exit_code == 0, explained.output
    payload = json.loads(explained.stdout)
    assert payload["field_accepted"] is True
    assert payload["field_effective"] is False
    assert payload["classification"] == "informational_only"

    config = Path(__file__).resolve().parents[2] / "examples/verl-opd-v0.8-single-gpu.yaml"
    checked = runner.invoke(
        app,
        [
            "compat",
            "check",
            "--profile",
            PROFILE,
            "--config",
            str(config),
            "--accept-local-reinterpretations",
            "--json",
        ],
    )
    assert checked.exit_code == 0, checked.output
    report = json.loads(checked.stdout)
    assert report["status"] == "compatible"
    assert report["summary"]["field_unsupported"] == 0


def test_profile_commands_remain_available_without_torch(monkeypatch) -> None:
    import miniverl.utils.lazy as lazy

    real_have_module = lazy.have_module
    monkeypatch.setattr(
        lazy, "have_module", lambda name: False if name == "torch" else real_have_module(name)
    )
    result = runner.invoke(app, ["profiles", "list", "--json"])
    assert result.exit_code == 0, result.output
    assert PROFILE in result.stdout
