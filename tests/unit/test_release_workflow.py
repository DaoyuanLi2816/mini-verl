"""Static safety assertions for the v0.2 release supply chain."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

WORKFLOW = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "release.yml"
GPU_WORKFLOW = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "gpu.yml"
EXISTING_RELEASE_WORKFLOW = (
    Path(__file__).resolve().parents[2] / ".github" / "workflows" / "verify-existing-release.yml"
)


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


def test_release_consumes_qualified_candidate_without_rebuild() -> None:
    text = _workflow_text()
    assert "python -m build" not in text
    prepare = _job(text, "prepare-verified-distributions", "publish-pypi")
    publish = _job(text, "publish-pypi", "verify-pypi")
    verify = _job(text, "verify-pypi", "create-github-release")
    release = _job(text, "create-github-release", None)

    assert "scripts/prepare_release_assets.py" in prepare
    assert "--check" in prepare
    assert 'item["name"].replace("_", "-")' not in prepare
    assert "candidate-manifest.json" in prepare
    assert "qualification.json" in prepare
    assert "qualification-SHA256SUMS" in prepare
    assert "actions/upload-artifact@" in prepare
    assert "actions/download-artifact@" in publish
    assert "python -m build" not in publish
    assert "actions/checkout@" in publish
    assert "scripts/check_pypi_publish_state.py" in publish
    assert "skip-existing" not in publish
    assert "steps.pypi-state.outputs.publish_needed == 'true'" in publish
    assert "actions/download-artifact@" in verify
    assert "actions/download-artifact@" in release
    assert text.count("name: release-distributions") == 4
    assert "needs: verify-pypi" in release


def test_every_action_is_pinned_to_a_full_commit_sha() -> None:
    text = (
        _workflow_text()
        + GPU_WORKFLOW.read_text(encoding="utf-8")
        + EXISTING_RELEASE_WORKFLOW.read_text(encoding="utf-8")
    )
    uses = re.findall(r"^\s*-\s+uses:\s+([^\s#]+)", text, flags=re.MULTILINE)
    assert uses
    for action in uses:
        assert re.fullmatch(r"[^@]+@[0-9a-f]{40}", action), action


def test_release_requires_same_run_exact_sha_candidate_and_qualification() -> None:
    text = _workflow_text()
    qualification = _job(text, "verify-qualified-candidate", "validate-and-test")
    validation = _job(text, "validate-and-test", "prepare-verified-distributions")
    prepare = _job(text, "prepare-verified-distributions", "publish-pypi")

    assert "permissions:" in qualification and "actions: read" in qualification
    assert "scripts/verify_release_qualification.py" in qualification
    assert '--commit "$GITHUB_SHA"' in qualification
    assert "--candidate-artifact-name candidate-distributions" in qualification
    assert "--qualification-artifact-name gpu-full-qualification" in qualification
    assert "--required-level full_qualification" in qualification
    assert '--required-gpu-name "NVIDIA GeForce RTX 4080"' in qualification
    assert "needs: verify-qualified-candidate" in validation
    assert "needs: [verify-qualified-candidate, validate-and-test]" in prepare


def test_gpu_workflow_builds_candidate_only_on_hosted_job() -> None:
    text = GPU_WORKFLOW.read_text(encoding="utf-8")
    build = _job(text, "build-candidate", "qualify")
    qualify = _job(text, "qualify", None)
    assert "runs-on: ubuntu-latest" in build
    assert "scripts/build_release_candidate.py" in build
    assert "name: candidate-distributions" in build
    assert "needs: build-candidate" in qualify
    assert "actions/download-artifact@" in qualify
    assert "python -m build" not in qualify
    assert '--candidate-manifest "$RUNNER_TEMP/candidate/candidate-manifest.json"' in qualify
    assert 'PYTHONPATH: ""' in qualify and 'PYTHONHOME: ""' in qualify
    assert "runs-on: [self-hosted, cuda, rtx4080, wsl2]" in qualify
    assert "shell: pwsh" not in qualify
    assert 'uv" venv --seed --python 3.12.13 "$RUNNER_TEMP/gpu-qualification-venv"' in qualify
    assert 'Path(os.environ["RUNNER_TEMP"]) / "gpu-qualification-venv/bin/python"' in qualify
    assert "Path('.gpu-qualification-venv/bin/python').resolve()" not in qualify
    assert "path: ${{ runner.temp }}/candidate" in qualify
    assert qualify.count("path: ${{ runner.temp }}/qualification/") == 2
    assert 'test -z "$(git status --porcelain)"' in qualify
    for name in (
        "candidate",
        "gpu-qualification-venv",
        "qualification",
        "qualification-work",
        "qualification-full",
        "qualification-full-work",
    ):
        assert f"$RUNNER_TEMP/{name}" in qualify


def test_gpu_workflow_refuses_rerun_attempts_and_separates_smoke_from_full() -> None:
    text = GPU_WORKFLOW.read_text(encoding="utf-8")
    assert text.count("GITHUB_RUN_ATTEMPT") >= 2
    assert text.count("start a new workflow_dispatch run") == 2
    smoke_upload = text.index("name: gpu-release-smoke")
    promotion = text.index("scripts/promote_full_gpu_qualification.py")
    full_upload = text.index("name: gpu-full-qualification")
    assert smoke_upload < promotion < full_upload
    assert "always() && inputs.qualification_level == 'full'" not in text


def test_full_gpu_qualification_installs_only_the_exact_pinned_verl_source() -> None:
    text = GPU_WORKFLOW.read_text(encoding="utf-8")
    requirement = (
        "git+https://github.com/verl-project/verl.git@7aed6b230776f963fa09509c10d9c3a767d1102c"
    )
    assert "qualification_level == 'full'" in text
    assert "pip install --no-deps" in text
    assert requirement in text
    assert "scripts/promote_full_gpu_qualification.py" in text
    for required in (
        "scripts/run_v011_profile_qualification.py",
        "--backend hf_cached",
        "--backend vllm",
        "--v011-profiles",
        "--hf-cached-runtime",
        "--vllm-runtime",
        "known-good-rtx4080-wsl2-cu130.txt",
    ):
        assert required in text


def test_release_accepts_only_the_v011_wsl2_known_good_stack() -> None:
    qualification = _job(_workflow_text(), "verify-qualified-candidate", "validate-and-test")
    assert "known-good-rtx4080-wsl2-cu130.json" in qualification
    assert "known-good-rtx4080-cu130.json" not in qualification


def test_release_recovery_preserves_full_qualification_evidence() -> None:
    text = EXISTING_RELEASE_WORKFLOW.read_text(encoding="utf-8")
    assert "release-artifacts/dist" in text
    assert text.count("qualification-full-SHA256SUMS") >= 3
    assert "release-artifacts/qualification-evidence/*" in text


def test_github_release_uses_only_explicit_canonical_assets() -> None:
    release = _job(_workflow_text(), "create-github-release", None)
    assert "release-artifacts/dist/*" not in release
    assert "release-artifacts/qualification-evidence/*" not in release
    for name in (
        "qualification-release-smoke.json",
        "qualification-direct-gkd.json",
        "qualification-pg-k1.json",
        "qualification-smollm2.json",
        "qualification-evidence.tar.gz",
        "qualification-evidence-manifest.json",
        "release-verification.json",
    ):
        assert f"release-artifacts/{name}" in release
