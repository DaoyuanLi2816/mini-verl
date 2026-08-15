from __future__ import annotations

import json
from pathlib import Path

import pytest


def _candidate(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    from miniverl.release_candidate import PINNED_BUILD_TOOLS, sha256_file

    root = tmp_path / "candidate"
    root.mkdir()
    wheel = root / "miniverl-0.10.1.dev0-py3-none-any.whl"
    sdist = root / "miniverl-0.10.1.dev0.tar.gz"
    wheel.write_bytes(b"wheel-one")
    sdist.write_bytes(b"sdist-one")
    payload: dict[str, object] = {
        "schema_version": 1,
        "kind": "miniverl_release_candidate",
        "source_commit": "a" * 40,
        "miniverl_version": "0.10.1.dev0",
        "created_at": "2026-08-15T12:00:00Z",
        "artifact_name": "candidate-distributions",
        "workflow": {
            "kind": "github_actions",
            "repository": "DaoyuanLi2816/mini-verl",
            "workflow_path": ".github/workflows/gpu.yml",
            "run_id": 42,
            "run_attempt": 1,
        },
        "build": {"os": "Linux", "python": "3.12.0", "tools": PINNED_BUILD_TOOLS},
        "wheel": {
            "filename": wheel.name,
            "bytes": wheel.stat().st_size,
            "sha256": sha256_file(wheel),
        },
        "sdist": {
            "filename": sdist.name,
            "bytes": sdist.stat().st_size,
            "sha256": sha256_file(sdist),
        },
    }
    (root / "candidate-manifest.json").write_text(
        json.dumps(payload, sort_keys=True), encoding="utf-8"
    )
    (root / "SHA256SUMS").write_text(
        f"{payload['wheel']['sha256']}  {wheel.name}\n"  # type: ignore[index]
        f"{payload['sdist']['sha256']}  {sdist.name}\n",  # type: ignore[index]
        encoding="utf-8",
    )
    return root, payload


def test_candidate_directory_is_exact_and_bound(tmp_path: Path) -> None:
    from miniverl.release_candidate import validate_candidate_directory

    root, _ = _candidate(tmp_path)
    assert (
        validate_candidate_directory(
            root,
            expected_commit="a" * 40,
            expected_version="0.10.1.dev0",
            expected_repository="DaoyuanLi2816/mini-verl",
            expected_workflow_path=".github/workflows/gpu.yml",
            expected_run_id=42,
            expected_run_attempt=1,
        )
        == []
    )


@pytest.mark.parametrize("mutation", ["wheel", "extra", "sums", "private", "nan"])
def test_candidate_rejects_tampering_extra_files_and_private_data(
    tmp_path: Path, mutation: str
) -> None:
    from miniverl.release_candidate import validate_candidate_directory

    root, payload = _candidate(tmp_path)
    if mutation == "wheel":
        (root / payload["wheel"]["filename"]).write_bytes(b"different")  # type: ignore[index]
    elif mutation == "extra":
        (root / "old.whl").write_bytes(b"old")
    elif mutation == "sums":
        (root / "SHA256SUMS").write_text("wrong\n", encoding="utf-8")
    else:
        if mutation == "private":
            payload["build"]["os"] = "C:\\Users\\maintainer\\OneDrive"  # type: ignore[index]
        else:
            payload["build"]["python"] = float("nan")  # type: ignore[index]
        (root / "candidate-manifest.json").write_text(
            json.dumps(payload, allow_nan=True), encoding="utf-8"
        )
    assert validate_candidate_directory(root)


def test_candidate_rejects_wrong_workflow_attempt(tmp_path: Path) -> None:
    from miniverl.release_candidate import validate_candidate_directory

    root, _ = _candidate(tmp_path)
    problems = validate_candidate_directory(root, expected_run_attempt=2)
    assert any("workflow run attempt" in problem for problem in problems)


def test_candidate_rejects_wrong_source_version_and_path_escape(tmp_path: Path) -> None:
    from miniverl.release_candidate import validate_candidate_directory

    root, payload = _candidate(tmp_path)
    problems = validate_candidate_directory(
        root, expected_commit="b" * 40, expected_version="0.10.1"
    )
    assert any("source commit" in problem for problem in problems)
    assert any("version" in problem for problem in problems)
    payload["wheel"]["filename"] = "../escape.whl"  # type: ignore[index]
    (root / "candidate-manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    assert validate_candidate_directory(root)


def test_candidate_rejects_symlink_when_supported(tmp_path: Path) -> None:
    from miniverl.release_candidate import validate_candidate_directory

    root, payload = _candidate(tmp_path)
    wheel = root / payload["wheel"]["filename"]  # type: ignore[index]
    target = root / "outside"
    target.write_bytes(wheel.read_bytes())
    wheel.unlink()
    try:
        wheel.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation needs privileges")
    assert validate_candidate_directory(root)


def test_candidate_builder_refuses_nonempty_output(tmp_path: Path) -> None:
    from miniverl.release_candidate import build_release_candidate

    output = tmp_path / "candidate"
    output.mkdir()
    (output / "old.whl").write_bytes(b"old")
    with pytest.raises(ValueError, match="empty"):
        build_release_candidate(output, source_commit="a" * 40, project_root=Path.cwd())
