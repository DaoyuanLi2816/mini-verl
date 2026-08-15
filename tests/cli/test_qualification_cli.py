from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from miniverl.cli import app
from tests.unit.test_gpu_qualification import _payload


def test_qualification_command_has_help_and_validates_json(tmp_path: Path) -> None:
    runner = CliRunner()
    help_result = runner.invoke(app, ["qualification", "validate", "--help"])
    assert help_result.exit_code == 0
    assert "exact source SHA" in help_result.output

    payload, root = _payload(tmp_path)
    path = root / "qualification.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    result = runner.invoke(
        app,
        [
            "qualification",
            "validate",
            str(path),
            "--commit",
            "a" * 40,
            "--required-gpu-name",
            "NVIDIA GeForce RTX 4080",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {"valid": True, "problems": []}


def test_qualification_command_fails_closed_on_wrong_commit(tmp_path: Path) -> None:
    payload, root = _payload(tmp_path)
    path = root / "qualification.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    result = CliRunner().invoke(
        app,
        ["qualification", "validate", str(path), "--commit", "f" * 40, "--json"],
    )
    assert result.exit_code == 1
    body = json.loads(result.stdout)
    assert body["valid"] is False
    assert body["problems"][0].startswith("binding:")
