"""The bundle tree is validated before any bundle content is read.

An exported bundle arrives from someone else. Every content check -- hashing,
reward parsing, YAML/JSON loading, the text metadata privacy scan -- opens paths
inside it, and an open follows whatever the path resolves to. Before v0.7.0 a
bundle could therefore point an entry at a file outside itself and have the
doctor read it.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from miniverl.bridge.preflight import preflight_bundle_tree


def _reasons(check: dict) -> set[str]:
    return {item["reason"] for item in check["rejections"]}


def _plain_bundle(root: Path) -> Path:
    (root / "model").mkdir(parents=True)
    (root / "provenance").mkdir(parents=True)
    (root / "model" / "adapter_config.json").write_text("{}", encoding="utf-8")
    (root / "provenance" / "SHA256SUMS").write_text("", encoding="utf-8")
    (root / "README.md").write_text("bundle", encoding="utf-8")
    return root


def _can_symlink(tmp_path: Path) -> bool:
    probe = tmp_path / "__symlink_probe"
    target = tmp_path / "__symlink_target"
    target.write_text("x", encoding="utf-8")
    try:
        probe.symlink_to(target)
    except (OSError, NotImplementedError):
        return False
    probe.unlink()
    return True


# ------------------------------------------------------------------ accepted


def test_a_plain_bundle_passes(tmp_path: Path) -> None:
    check = preflight_bundle_tree(_plain_bundle(tmp_path / "bundle"))

    assert check["status"] == "ok"
    assert check["files"] == 3
    assert check["rejections"] == []
    assert check["nominal_bytes"] > 0


def test_an_empty_directory_tree_passes(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    (root / "data").mkdir(parents=True)

    check = preflight_bundle_tree(root)

    assert check["status"] == "ok"
    assert check["files"] == 0


# ------------------------------------------------------------------ rejected


def test_a_file_symlink_is_rejected(tmp_path: Path) -> None:
    if not _can_symlink(tmp_path):
        pytest.skip("symlink creation needs privileges")
    secret = tmp_path / "outside-secret.txt"
    secret.write_text("api_key=FAKE_NOT_A_REAL_SECRET", encoding="utf-8")
    root = _plain_bundle(tmp_path / "bundle")
    (root / "model" / "tokenizer_config.json").symlink_to(secret)

    check = preflight_bundle_tree(root)

    assert check["status"] == "fail"
    assert "symlink" in _reasons(check)


def test_a_directory_symlink_is_rejected(tmp_path: Path) -> None:
    if not _can_symlink(tmp_path):
        pytest.skip("symlink creation needs privileges")
    outside = tmp_path / "outside-dir"
    outside.mkdir()
    (outside / "secret.txt").write_text("token=FAKE_NOT_A_REAL_SECRET", encoding="utf-8")
    root = _plain_bundle(tmp_path / "bundle")
    (root / "data").symlink_to(outside, target_is_directory=True)

    check = preflight_bundle_tree(root)

    assert check["status"] == "fail"
    assert "symlink" in _reasons(check)


def test_a_symlink_that_stays_inside_the_bundle_is_still_rejected(tmp_path: Path) -> None:
    """The rule is structural. An inside-pointing link is still a redirection."""
    if not _can_symlink(tmp_path):
        pytest.skip("symlink creation needs privileges")
    root = _plain_bundle(tmp_path / "bundle")
    (root / "alias.md").symlink_to(root / "README.md")

    check = preflight_bundle_tree(root)

    assert check["status"] == "fail"
    assert "symlink" in _reasons(check)


@pytest.mark.skipif(sys.platform != "win32", reason="junctions are Windows-specific")
def test_a_windows_junction_is_rejected(tmp_path: Path) -> None:
    outside = tmp_path / "outside-dir"
    outside.mkdir()
    (outside / "secret.txt").write_text("password=FAKE_NOT_A_REAL_SECRET", encoding="utf-8")
    root = _plain_bundle(tmp_path / "bundle")
    junction = root / "data"
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(outside)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip(f"mklink /J unavailable: {result.stderr.strip()}")

    check = preflight_bundle_tree(root)

    assert check["status"] == "fail"
    # A junction is a reparse point; Python may or may not also call it a link.
    assert _reasons(check) & {"reparse_point", "symlink"}


@pytest.mark.skipif(sys.platform == "win32", reason="FIFOs are POSIX-specific")
def test_a_fifo_is_rejected(tmp_path: Path) -> None:
    root = _plain_bundle(tmp_path / "bundle")
    os.mkfifo(root / "model" / "adapter_model.safetensors")

    check = preflight_bundle_tree(root)

    assert check["status"] == "fail"
    assert "non_regular_file" in _reasons(check)


def test_the_file_count_is_bounded(tmp_path: Path) -> None:
    root = _plain_bundle(tmp_path / "bundle")
    for index in range(12):
        (root / f"pad-{index}.txt").write_text("x", encoding="utf-8")

    check = preflight_bundle_tree(root, max_files=8)

    assert check["status"] == "fail"
    assert "max_files_exceeded" in _reasons(check)


def test_the_nominal_byte_total_is_bounded(tmp_path: Path) -> None:
    root = _plain_bundle(tmp_path / "bundle")
    (root / "big.bin").write_bytes(b"\0" * 4096)

    check = preflight_bundle_tree(root, max_nominal_bytes=1024)

    assert check["status"] == "fail"
    assert "max_nominal_bytes_exceeded" in _reasons(check)


def test_depth_is_bounded(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    deep = root
    for index in range(8):
        deep = deep / f"level{index}"
    deep.mkdir(parents=True)

    check = preflight_bundle_tree(root, max_depth=3)

    assert check["status"] == "fail"
    assert "max_depth_exceeded" in _reasons(check)


def test_a_missing_root_fails_closed(tmp_path: Path) -> None:
    check = preflight_bundle_tree(tmp_path / "absent")

    assert check["status"] == "fail"
    assert "unresolvable_root" in _reasons(check)


def test_a_file_as_root_fails_closed(tmp_path: Path) -> None:
    target = tmp_path / "not-a-directory"
    target.write_text("x", encoding="utf-8")

    check = preflight_bundle_tree(target)

    assert check["status"] == "fail"
    assert "root_not_a_directory" in _reasons(check)


@pytest.mark.skipif(sys.platform != "win32", reason="junctions are Windows-specific")
def test_the_doctor_refuses_a_redirected_bundle_without_reading_outside_it(tmp_path: Path) -> None:
    """The whole point: a hostile bundle cannot make the doctor read your files.

    Before v0.7.0 this bundle got the outside file hashed and its content
    searched for credentials, with the detector category reported back.
    """
    from miniverl.bridge.doctor import inspect_bridge_bundle

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "credentials.json").write_text(
        '{"api_key": "FAKE_NOT_A_REAL_SECRET_0123456789"}', encoding="utf-8"
    )
    root = _plain_bundle(tmp_path / "bundle")
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(root / "data"), str(outside)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip(f"mklink /J unavailable: {result.stderr.strip()}")

    report = inspect_bridge_bundle(root)

    assert report["verdict"] == "fail"
    assert report["bundle_tree_preflight"]["status"] == "fail"
    # The checks did not run, so they report "not inspected" rather than a
    # finding they are not entitled to make.
    assert report["artifact_hashes"]["status"] == "not_inspected"
    assert report["portable_metadata_privacy"] == "not_inspected"
    assert report["privacy"]["status"] == "not_inspected"
    assert report["launchable"] is False


def test_the_preflight_never_reads_file_content(tmp_path: Path, monkeypatch) -> None:
    """Preflight runs before content checks, so it must not open a file."""
    root = _plain_bundle(tmp_path / "bundle")
    opened: list[str] = []
    real_open = Path.open

    def recording_open(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        opened.append(str(self))
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", recording_open)

    check = preflight_bundle_tree(root)

    assert check["status"] == "ok"
    assert opened == []
