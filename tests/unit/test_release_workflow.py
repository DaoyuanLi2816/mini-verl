"""Static safety assertions for the v0.2 release supply chain."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

WORKFLOW = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "release.yml"
GPU_WORKFLOW = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "gpu.yml"


def _workflow_text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def _job(text: str, name: str, next_name: str | None) -> str:
    start = text.index(f"  {name}:")
    end = text.index(f"  {next_name}:", start) if next_name else len(text)
    return text[start:end]


@pytest.mark.parametrize(
    ("event_name", "ref", "expected"),
    [
        ("workflow_dispatch", "refs/heads/main", False),
        ("workflow_dispatch", "refs/tags/v0.2.0", False),
        ("push", "refs/heads/main", False),
        ("push", "refs/tags/not-a-release", False),
        ("push", "refs/tags/v0.2.0", True),
    ],
)
def test_publish_eligibility_is_tag_push_only(
    event_name: str,
    ref: str,
    expected: bool,
) -> None:
    eligible = event_name == "push" and ref.startswith("refs/tags/v")
    assert eligible is expected
    if expected:
        condition = "github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v')"
        text = _workflow_text()
        assert text.count(condition) == 3


def test_distributions_are_built_once_and_consumed_without_rebuild() -> None:
    text = _workflow_text()
    assert text.count("python -m build") == 1
    build = _job(text, "build-distributions", "publish-pypi")
    publish = _job(text, "publish-pypi", "verify-pypi")
    verify = _job(text, "verify-pypi", "create-github-release")
    release = _job(text, "create-github-release", None)

    assert "python -m build" in build
    assert "twine check" in build
    assert "SHA256SUMS" in build
    assert "actions/upload-artifact@" in build
    assert "actions/download-artifact@" in publish
    assert "python -m build" not in publish
    assert "actions/checkout@" not in publish
    assert "actions/download-artifact@" in verify
    assert "actions/download-artifact@" in release
    assert text.count("name: release-distributions") == 4
    assert "needs: verify-pypi" in release


def test_every_action_is_pinned_to_a_full_commit_sha() -> None:
    uses = re.findall(r"^\s*-\s+uses:\s+([^\s#]+)", _workflow_text(), flags=re.MULTILINE)
    assert uses
    for action in uses:
        assert re.fullmatch(r"[^@]+@[0-9a-f]{40}", action), action


def test_release_requires_exact_sha_rtx4080_qualification_before_build() -> None:
    text = _workflow_text()
    qualification = _job(text, "verify-gpu-qualification", "validate-and-test")
    validation = _job(text, "validate-and-test", "build-distributions")
    build = _job(text, "build-distributions", "publish-pypi")

    assert "permissions:" in qualification and "actions: read" in qualification
    assert "scripts/verify_release_qualification.py" in qualification
    assert '--commit "$GITHUB_SHA"' in qualification
    assert '--required-gpu-name "NVIDIA GeForce RTX 4080"' in qualification
    assert "needs: verify-gpu-qualification" in validation
    assert "needs: validate-and-test" in build


def test_full_gpu_qualification_installs_only_the_exact_pinned_verl_source() -> None:
    text = GPU_WORKFLOW.read_text(encoding="utf-8")
    requirement = (
        "git+https://github.com/verl-project/verl.git@7aed6b230776f963fa09509c10d9c3a767d1102c"
    )
    assert "qualification_level == 'full'" in text
    assert "pip install --no-deps" in text
    assert requirement in text
    assert "scripts/promote_full_gpu_qualification.py" in text
