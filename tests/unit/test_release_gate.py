from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

from scripts import release_gate


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
        "candidate_artifact",
        "twine",
        "release_evidence_chain",
    ):
        assert required in names
    text = Path("scripts/release_gate.py").read_text(encoding="utf-8")
    assert "gh release create" not in text
    assert "git tag" not in text
    assert "twine upload" not in text


def test_release_gate_list_does_not_require_a_git_checkout(tmp_path: Path) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copy2("scripts/release_gate.py", scripts / "release_gate.py")

    completed = subprocess.run(
        [sys.executable, "scripts/release_gate.py", "--list"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "release_evidence_chain" in json.loads(completed.stdout)


def test_release_gate_requires_qualification_when_running() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/release_gate.py", "--fail-fast"],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "--candidate-dir, --candidate-manifest and --qualification" in completed.stderr
    assert "required unless --list is used" in completed.stderr


def test_release_gate_binds_the_wsl2_known_good_environment(tmp_path: Path) -> None:
    plan = release_gate.gate_plan(
        candidate_dir=tmp_path / "candidate",
        candidate_wheel=tmp_path / "candidate.whl",
        candidate_sdist=tmp_path / "candidate.tar.gz",
        qualification=tmp_path / "qualification.json",
        site=tmp_path / "site",
        screenshots=tmp_path / "screenshots",
        source_commit="a" * 40,
    )
    command = next(gate.command for gate in plan if gate.name == "release_evidence_chain")
    known_good_sha256 = command[command.index("--known-good-sha256") + 1]

    wsl2_manifest = Path("environments/known-good-rtx4080-wsl2-cu130.json")
    legacy_manifest = Path("environments/known-good-rtx4080-cu130.json")
    assert known_good_sha256 == hashlib.sha256(wsl2_manifest.read_bytes()).hexdigest()
    assert known_good_sha256 != hashlib.sha256(legacy_manifest.read_bytes()).hexdigest()
