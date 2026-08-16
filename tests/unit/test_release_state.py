"""Stable and development version claims have exactly one source.

After v0.6.2 shipped, the README still said stable was v0.6.1, the docs selector
offered "Stable 0.6.1 / Development 0.6.2.dev0", and the quality record paired
``release: 0.6.2`` with a floor measured "at v0.6.1". Each was an independently
hand-edited literal. ``release-state.yaml`` is now the only source and this gate
proves every projection of it agrees.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest

import miniverl
from scripts.release_state import (
    ReleaseState,
    apply_release_state,
    check_release_state,
    load_release_state,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Every file the canonical state projects onto.
TRACKED = (
    "release-state.yaml",
    "src/miniverl/__init__.py",
    "README.md",
    "README.zh-CN.md",
    "PYPI.md",
    "docs/overrides/main.html",
    "CITATION.cff",
    "SECURITY.md",
    "CHANGELOG.md",
    "docs/release-checklist.md",
    "PROJECT_STATE.md",
    "docs/generated/quality.json",
)


# ------------------------------------------------------------ the live gate


def test_repository_agrees_with_the_canonical_release_state() -> None:
    assert check_release_state(REPO_ROOT) == []


def test_package_version_is_the_declared_development_version() -> None:
    state = load_release_state(REPO_ROOT)
    assert miniverl.__version__ == state.development_version


def test_docs_selector_matches_both_channels() -> None:
    state = load_release_state(REPO_ROOT)
    html = (REPO_ROOT / "docs" / "overrides" / "main.html").read_text(encoding="utf-8")
    assert f'data-stable-version="{state.stable_version}"' in html
    assert f'data-dev-version="{state.development_version}"' in html
    assert f">Stable {state.stable_version}<" in html
    assert f">Development {state.development_version}<" in html


def test_quality_record_floor_names_its_own_release() -> None:
    record = json.loads(
        (REPO_ROOT / "docs" / "generated" / "quality.json").read_text(encoding="utf-8")
    )
    assert f"v{record['release']}" in record["quality_floor"]


def test_security_and_current_product_prose_follow_the_canonical_state() -> None:
    state = load_release_state(REPO_ROOT)
    security = (REPO_ROOT / "SECURITY.md").read_text(encoding="utf-8")
    project = (REPO_ROOT / "PROJECT_STATE.md").read_text(encoding="utf-8")
    assert f"supported stable line is `{state.stable_version}`" in security
    assert f"Development `{state.development_version}` has a closed typed profile" in project


# --------------------------------------------------------------- validation


def test_tag_must_match_the_version() -> None:
    with pytest.raises(ValueError, match=re.escape("must be v0.6.3")):
        ReleaseState("0.6.3", "v0.6.2", "a" * 40, "2026-08-05", "0.6.4.dev0").validate()


def test_development_version_must_be_a_dev_version() -> None:
    with pytest.raises(ValueError, match=re.escape("must end in .devN")):
        ReleaseState("0.6.3", "v0.6.3", "a" * 40, "2026-08-05", "0.6.4").validate()


def test_development_version_may_not_re_release_the_stable_version() -> None:
    with pytest.raises(ValueError, match="re-release"):
        ReleaseState("0.6.3", "v0.6.3", "a" * 40, "2026-08-05", "0.6.3.dev1").validate()


def test_release_commit_must_be_a_full_sha() -> None:
    with pytest.raises(ValueError, match="40-character"):
        ReleaseState("0.6.3", "v0.6.3", "bef9f08", "2026-08-05", "0.6.4.dev0").validate()


def test_preparing_version_strips_the_dev_suffix() -> None:
    state = ReleaseState("0.6.2", "v0.6.2", "a" * 40, "2026-08-05", "0.6.3.dev0")
    assert state.preparing_version == "0.6.3"


# ------------------------------------------------- post-release transition


def _clone(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    for relative in TRACKED:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / relative, destination)
    return root


def test_publishing_0_6_3_then_syncing_main_updates_every_claim(tmp_path: Path) -> None:
    """release 0.6.3 -> stable 0.6.3 -> main 0.6.4.dev0, with no manual editing."""
    root = _clone(tmp_path)
    released = ReleaseState(
        stable_version="0.6.3",
        stable_tag="v0.6.3",
        stable_commit="b" * 40,
        stable_released_at="2026-08-06",
        development_version="0.6.4.dev0",
    )

    apply_release_state(root, released)

    readme = (root / "README.md").read_text(encoding="utf-8")
    assert "PyPI `v0.6.3` is stable" in readme
    chinese = (root / "README.zh-CN.md").read_text(encoding="utf-8")
    assert "PyPI `v0.6.3` 是稳定版" in chinese
    html = (root / "docs" / "overrides" / "main.html").read_text(encoding="utf-8")
    assert 'data-stable-version="0.6.3"' in html
    assert 'data-dev-version="0.6.4.dev0"' in html
    assert ">Stable 0.6.3<" in html
    assert ">Development 0.6.4.dev0<" in html
    assert '__version__ = "0.6.4.dev0"' in (root / "src" / "miniverl" / "__init__.py").read_text(
        encoding="utf-8"
    )
    citation = (root / "CITATION.cff").read_text(encoding="utf-8")
    assert "version: 0.6.3" in citation
    assert "date-released: 2026-08-06" in citation
    changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "compare/v0.6.3...HEAD" in changelog


def test_transition_still_demands_the_human_written_sections(tmp_path: Path) -> None:
    """The gate must not silently pass an incomplete state sync.

    Uses the release *after* the one the checked-in files describe, so the
    checklist section, the PROJECT_STATE line and the quality record are all
    genuinely stale rather than accidentally still correct.
    """
    root = _clone(tmp_path)
    released = ReleaseState(
        stable_version="0.6.4",
        stable_tag="v0.6.4",
        stable_commit="b" * 40,
        stable_released_at="2026-09-01",
        development_version="0.6.5.dev0",
    )
    apply_release_state(root, released)

    problems = check_release_state(root, released)

    # The checklist section and the PROJECT_STATE line are prose a person writes.
    assert any("release-checklist" in problem for problem in problems)
    assert any("PROJECT_STATE" in problem for problem in problems)
    assert any("quality.json" in problem for problem in problems)

    # Supplying them clears the gate without touching anything else.
    checklist = root / "docs" / "release-checklist.md"
    checklist.write_text("# Release checklist\n\n## v0.6.5 next\n", encoding="utf-8")
    state_file = root / "PROJECT_STATE.md"
    state_file.write_text(
        "# PROJECT_STATE\n\nCanonical release state: stable `v0.6.4` "
        f"(`{'b' * 40}`), development `0.6.5.dev0`.\n\n"
        "Development `0.6.5.dev0` has a closed typed profile registry.\n",
        encoding="utf-8",
    )
    record = root / "docs" / "generated" / "quality.json"
    payload = json.loads(record.read_text(encoding="utf-8"))
    payload["release"] = "0.6.4"
    payload["quality_floor"] = "1,560+ tests and 85%+ branch coverage at v0.6.4"
    record.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    assert check_release_state(root, released) == []


def test_release_phase_treats_the_tree_as_the_release(tmp_path: Path) -> None:
    """A release tree *is* the release, so both channels name the same version.

    v0.6.2 had no release phase, which is how its tag shipped a docs selector
    still advertising "Stable 0.6.1 / Development 0.6.2.dev0".
    """
    root = _clone(tmp_path)
    releasing = ReleaseState(
        stable_version="0.6.3",
        stable_tag="v0.6.3",
        stable_commit="pending",
        stable_released_at="2026-08-06",
        development_version="0.6.3",
        phase="release",
    )

    apply_release_state(root, releasing)

    assert '__version__ = "0.6.3"' in (root / "src" / "miniverl" / "__init__.py").read_text(
        encoding="utf-8"
    )
    html = (root / "docs" / "overrides" / "main.html").read_text(encoding="utf-8")
    assert 'data-stable-version="0.6.3"' in html
    assert 'data-dev-version="0.6.3"' in html
    assert "PyPI `v0.6.3` is stable" in (root / "README.md").read_text(encoding="utf-8")


def test_release_phase_rejects_a_dev_version(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must equal the version being published"):
        ReleaseState(
            "0.6.3", "v0.6.3", "pending", "2026-08-06", "0.6.3.dev0", phase="release"
        ).validate()


def test_pending_commit_is_only_accepted_while_releasing() -> None:
    ReleaseState("0.6.3", "v0.6.3", "pending", "2026-08-06", "0.6.3", phase="release").validate()
    with pytest.raises(ValueError, match="40-character"):
        ReleaseState(
            "0.6.3", "v0.6.3", "pending", "2026-08-06", "0.6.4.dev0", phase="development"
        ).validate()


def test_unknown_phase_is_rejected() -> None:
    with pytest.raises(ValueError, match="phase must be one of"):
        ReleaseState("0.6.3", "v0.6.3", "a" * 40, "2026-08-06", "0.6.4.dev0", phase="rc").validate()


def test_stale_claim_is_reported_with_the_expected_value(tmp_path: Path) -> None:
    """The shape of the v0.6.2 drift, reproduced against the gate.

    The stale value is derived from the canonical state rather than hard-coded,
    so cutting a release does not silently retarget this test.
    """
    root = _clone(tmp_path)
    state = load_release_state(root)
    readme = root / "README.md"
    readme.write_text(
        readme.read_text(encoding="utf-8").replace(
            f"PyPI `v{state.stable_version}` is stable", "PyPI `v0.0.1` is stable"
        ),
        encoding="utf-8",
    )

    problems = check_release_state(root)

    assert any(f"'0.0.1', expected '{state.stable_version}'" in problem for problem in problems)
