"""Prevent v1-readiness prose from drifting behind canonical release state."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_v1_readiness_matches_canonical_v0101_release_evidence() -> None:
    state = yaml.safe_load((ROOT / "release-state.yaml").read_text(encoding="utf-8"))
    text = (ROOT / "docs/v1-readiness.md").read_text(encoding="utf-8")
    stable = state["stable"]
    assert stable["version"] >= "0.10.1"
    assert stable["release_commit"] in text
    assert "passed in v0.10.1" in text
    assert "31932226695" in text
    assert "31933844796" in text
    assert "31934196365" in text


def test_v1_readiness_has_no_obsolete_or_contradictory_gate() -> None:
    text = (ROOT / "docs/v1-readiness.md").read_text(encoding="utf-8").lower()
    obsolete = (
        "private-runner activation",
        "private runner activation",
        "first successful same-run",
        "first same-run pair",
    )
    assert all(phrase not in text for phrase in obsolete)
    assert "not a v1 candidate" in text
    assert "not yet satisfied" in text
    assert "v1 public api baseline" in text
    assert "backward-compatibility evidence" in text
    assert "independent reproduction or explicit v1 hardware scope" in text
    assert "dedicated v1 candidate" in text


def test_polish_does_not_change_version_or_beta_classifier() -> None:
    state = yaml.safe_load((ROOT / "release-state.yaml").read_text(encoding="utf-8"))
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert state["stable"]["version"] == "0.10.1"
    assert state["development"]["version"] == "0.11.0.dev0"
    assert "Development Status :: 4 - Beta" in pyproject
