from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_release_gate_lists_the_product_checks_without_publishing(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/release_gate.py",
            "--list",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    names = json.loads(completed.stdout)
    for required in (
        "release_state",
        "known_good_environment",
        "ruff_check",
        "mypy",
        "actionlint",
        "cpu_tests_coverage",
        "docs_build",
        "docs_visual",
        "build",
        "twine",
        "gpu_qualification",
    ):
        assert required in names
    text = Path("scripts/release_gate.py").read_text(encoding="utf-8")
    assert "gh release create" not in text
    assert "git tag" not in text
    assert "twine upload" not in text


def test_release_gate_requires_qualification_when_running() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/release_gate.py", "--fail-fast"],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "--qualification is required unless --list is used" in completed.stderr
