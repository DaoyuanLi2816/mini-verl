from __future__ import annotations

import json
from pathlib import Path

import pytest


def _manifest(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    expected = {
        "miniverl-0.10.1-py3-none-any.whl": "a" * 64,
        "miniverl-0.10.1.tar.gz": "b" * 64,
    }
    payload = {
        "kind": "miniverl_release_candidate",
        "miniverl_version": "0.10.1",
        "wheel": {"filename": next(iter(expected)), "sha256": "a" * 64},
        "sdist": {"filename": list(expected)[1], "sha256": "b" * 64},
    }
    path = tmp_path / "candidate-manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, expected


def _response(files: dict[str, str]) -> dict[str, object]:
    return {
        "urls": [
            {"filename": filename, "digests": {"sha256": digest}}
            for filename, digest in files.items()
        ]
    }


def test_absent_pypi_version_needs_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import check_pypi_publish_state as state

    manifest, _ = _manifest(tmp_path)
    monkeypatch.setattr(state, "_request_version", lambda *args, **kwargs: None)
    assert state.publish_needed(manifest, project="miniverl") is True


def test_exact_existing_pypi_version_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import check_pypi_publish_state as state

    manifest, expected = _manifest(tmp_path)
    monkeypatch.setattr(state, "_request_version", lambda *args, **kwargs: _response(expected))
    assert state.publish_needed(manifest, project="miniverl") is False


@pytest.mark.parametrize(
    "mutation",
    [
        lambda files: files.pop("miniverl-0.10.1.tar.gz"),
        lambda files: files.update({"extra.zip": "c" * 64}),
        lambda files: files.update({"miniverl-0.10.1.tar.gz": "c" * 64}),
    ],
)
def test_existing_partial_extra_or_mismatched_version_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation
) -> None:
    from scripts import check_pypi_publish_state as state

    manifest, expected = _manifest(tmp_path)
    mutation(expected)
    monkeypatch.setattr(state, "_request_version", lambda *args, **kwargs: _response(expected))
    with pytest.raises(ValueError, match="does not match"):
        state.publish_needed(manifest, project="miniverl")
