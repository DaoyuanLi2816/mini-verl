from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.apply_docs_editorial_overlay import apply_overlay, normalized_bytes


def fixture(tmp_path: Path) -> tuple[Path, Path]:
    root, stable = tmp_path / "main", tmp_path / "stable"
    for directory in (root, stable):
        (directory / "docs").mkdir(parents=True)
    (root / "scripts").mkdir()
    entries = {}
    for name in ("index.md", "limitations.md"):
        (stable / "docs" / name).write_text("old\n", encoding="utf-8")
        (root / "docs" / name).write_text("new\n", encoding="utf-8")
        entries[f"docs/{name}"] = {
            "before": hashlib.sha256(b"old\n").hexdigest(),
            "after": hashlib.sha256(b"new\n").hexdigest(),
        }
    (root / "scripts/docs_editorial_overlay.json").write_text(
        json.dumps({"release_commit": "a" * 40, "files": entries}), encoding="utf-8"
    )
    return root, stable


def test_only_reviewed_pages_are_overlaid(tmp_path: Path) -> None:
    root, stable = fixture(tmp_path)
    evidence = stable / "docs/result.json"
    evidence.write_bytes(b'{"score":0}')
    assert apply_overlay(root, stable, "a" * 40) == 2
    assert normalized_bytes(stable / "docs/index.md") == b"new\n"
    assert evidence.read_bytes() == b'{"score":0}'


def test_other_releases_keep_their_own_docs(tmp_path: Path) -> None:
    root, stable = fixture(tmp_path)
    assert apply_overlay(root, stable, "b" * 40) == 0
    assert normalized_bytes(stable / "docs/index.md") == b"old\n"


@pytest.mark.parametrize("which", ["main", "stable"])
def test_any_drift_refuses_before_writing_a_page(tmp_path: Path, which: str) -> None:
    root, stable = fixture(tmp_path)
    target = root if which == "main" else stable
    (target / "docs/limitations.md").write_text("unreviewed", encoding="utf-8")
    with pytest.raises(ValueError, match="differs"):
        apply_overlay(root, stable, "a" * 40)
    assert normalized_bytes(stable / "docs/index.md") == b"old\n"


@pytest.mark.parametrize("path", ["", "../outside.md", "docs/result.json", "src/model.py"])
def test_overlay_cannot_touch_code_or_evidence(tmp_path: Path, path: str) -> None:
    root, stable = fixture(tmp_path)
    manifest = root / "scripts/docs_editorial_overlay.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["files"][path] = data["files"].pop("docs/index.md")
    manifest.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="path"):
        apply_overlay(root, stable, "a" * 40)
    assert normalized_bytes(stable / "docs/limitations.md") == b"old\n"


def test_git_checkout_line_endings_are_portable(tmp_path: Path) -> None:
    root, stable = fixture(tmp_path)
    (stable / "docs/index.md").write_bytes(b"old\r\n")
    (root / "docs/index.md").write_bytes(b"new\r\n")
    assert apply_overlay(root, stable, "a" * 40) == 2
