"""An incomplete privacy scan says so instead of reporting a clean bill.

The metadata scan is bounded: it skips files over a size limit, stops at a
total byte budget, and gives up on anything it cannot decode. Before v0.7.0 all
of those still produced ``heuristic_passed`` -- "I found nothing" reported for
an inspection that never looked. Nothing found over an incomplete scan is a
weaker statement and now has its own name.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from miniverl.bridge.doctor import (
    METADATA_MAX_FILE_BYTES,
    _check_privacy,
    _scan_portable_metadata,
)

SENTINEL = "FAKE_NOT_A_REAL_SECRET_0123456789"


def _bundle(root: Path) -> Path:
    (root / "provenance").mkdir(parents=True)
    (root / "README.md").write_text("a portable bundle", encoding="utf-8")
    (root / "provenance" / "miniverl-manifest.json").write_text(
        json.dumps({"schema_version": 1, "run_id": "fixture"}), encoding="utf-8"
    )
    return root


def _reasons(scan: dict[str, Any]) -> set[str]:
    return {item["reason"] for item in scan["incomplete_reasons"]}


# --------------------------------------------------------------------- states


def test_a_fully_inspected_clean_bundle_is_passed_full(tmp_path: Path) -> None:
    scan = _scan_portable_metadata(_bundle(tmp_path / "bundle"), sentinels=())

    assert scan["status"] == "heuristic_passed_full"
    assert scan["complete"] is True
    assert scan["incomplete_reasons"] == []


def test_a_finding_still_fails(tmp_path: Path) -> None:
    root = _bundle(tmp_path / "bundle")
    (root / "config.json").write_text(json.dumps({"api_key": SENTINEL}), encoding="utf-8")

    scan = _scan_portable_metadata(root, sentinels=())

    assert scan["status"] == "heuristic_failed"
    assert scan["findings"]
    assert not any(SENTINEL in json.dumps(item) for item in scan["findings"])


def test_a_file_skipped_for_size_reports_incomplete(tmp_path: Path) -> None:
    root = _bundle(tmp_path / "bundle")
    (root / "huge.md").write_text("x" * (METADATA_MAX_FILE_BYTES + 1), encoding="utf-8")

    scan = _scan_portable_metadata(root, sentinels=())

    # Nothing was found, but one file was never opened.
    assert scan["findings"] == []
    assert scan["status"] == "heuristic_incomplete"
    assert scan["complete"] is False
    assert "file_larger_than_scan_limit" in _reasons(scan)
    assert any(item["file"] == "huge.md" for item in scan["incomplete_reasons"])


def test_an_undecodable_file_reports_incomplete(tmp_path: Path) -> None:
    root = _bundle(tmp_path / "bundle")
    # A .txt the scanner will try to read as text but cannot decode.
    (root / "broken.txt").write_bytes(b"\xff\xfe\x00\x81\x8d\x8f")

    scan = _scan_portable_metadata(root, sentinels=())

    assert scan["findings"] == []
    assert scan["status"] == "heuristic_incomplete"
    assert "not_decodable_as_utf8" in _reasons(scan)


def test_incomplete_never_becomes_passed_in_the_privacy_report(tmp_path: Path) -> None:
    root = _bundle(tmp_path / "bundle")
    (root / "huge.md").write_text("x" * (METADATA_MAX_FILE_BYTES + 1), encoding="utf-8")

    privacy = _check_privacy(root)

    assert privacy["portable_metadata_privacy"] == "heuristic_incomplete"
    assert privacy["portable_metadata_scan_complete"] is False
    # Default behaviour: reported, not failed.
    assert privacy["status"] == "ok"


def test_strict_mode_fails_on_an_incomplete_scan(tmp_path: Path) -> None:
    root = _bundle(tmp_path / "bundle")
    (root / "huge.md").write_text("x" * (METADATA_MAX_FILE_BYTES + 1), encoding="utf-8")

    privacy = _check_privacy(root, require_complete_metadata_scan=True)

    assert privacy["portable_metadata_privacy"] == "heuristic_incomplete"
    assert privacy["strict_metadata_scan_required"] is True
    assert privacy["status"] == "fail"


def test_strict_mode_accepts_a_complete_clean_scan(tmp_path: Path) -> None:
    privacy = _check_privacy(_bundle(tmp_path / "bundle"), require_complete_metadata_scan=True)

    assert privacy["portable_metadata_privacy"] == "heuristic_passed_full"
    assert privacy["status"] == "ok"


def test_incomplete_reasons_are_bounded(tmp_path: Path) -> None:
    """A tree full of unreadable entries cannot flood the report."""
    root = _bundle(tmp_path / "bundle")
    for index in range(5):
        (root / f"broken-{index}.txt").write_bytes(b"\xff\xfe\x00\x81\x8d\x8f")

    scan = _scan_portable_metadata(root, sentinels=())

    assert scan["status"] == "heuristic_incomplete"
    assert len(scan["incomplete_reasons"]) == 5
    assert len(scan["incomplete_reasons"]) <= scan["max_findings"]
