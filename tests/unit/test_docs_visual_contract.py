from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_docs_use_the_pinned_modern_responsive_theme() -> None:
    config = yaml.safe_load((ROOT / "mkdocs.yml").read_text(encoding="utf-8"))
    assert config["theme"]["name"] == "material"
    assert "content.code.copy" in config["theme"]["features"]
    assert "navigation.footer" in config["theme"]["features"]
    assert config["plugins"] == ["search"]
    assert "assets/stylesheets/extra.css" in config["extra_css"]
    assert "assets/javascripts/versioning.js" in config["extra_javascript"]

    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "mkdocs-material==9.7.7" in pyproject
    assert "playwright==1.62.0" in pyproject


def test_versioned_docs_and_browser_visual_gate_are_wired() -> None:
    workflow = (ROOT / ".github/workflows/docs.yml").read_text(encoding="utf-8")
    assert "mkdocs build --strict" in workflow
    assert "site/dev" in workflow
    assert "playwright install --with-deps chromium" in workflow
    assert "docs-visual-screenshots" in workflow

    script = (ROOT / "scripts/check_docs_visual.py").read_text(encoding="utf-8")
    for viewport in ("1440, 900", "1024, 768", "820, 1000", "390, 844"):
        assert viewport in script
    for page in (
        '"/"',
        '"/alignment-lab/alignment-lab-v1/"',
        '"/consumer-runtime/"',
        '"/recoverybench/recoverybench-v1/"',
        '"/verl-bridge/"',
    ):
        assert page in script
    for contract in (
        "horizontal document overflow",
        "figure exceeds content column",
        "text outside the viewBox",
        "overlapping text",
        "label occludes a data mark",
        "table overflows without a scroll container",
        "responsive table problem",
        "mobile bridge did not select the vertical layout",
        "mobile alignment figures did not activate",
        "figure did not load",
        "text below",
    ):
        assert contract in script

    # The gate must measure real geometry, not a constant scale, and must not
    # restrict itself to authored [data-role] nodes.
    assert "svg.querySelectorAll('text')" in script
    assert "Math.min(820" not in script
    assert "currentSrc" in script
    assert "MIN_FONT_PX_NARROW = 11.0" in script


def test_the_mobile_readability_exemption_list_is_explicit_and_reported() -> None:
    """A bounded gate must name what it does not enforce."""
    script = (ROOT / "scripts/check_docs_visual.py").read_text(encoding="utf-8")
    assert "MOBILE_READABILITY_EXEMPT" in script
    assert "mobile readability NOT enforced" in script
    assert "unexpected figure claimed a mobile-readability exemption" in script
    for name in (
        "consumer-runtime-v1-pareto.svg",
        "cost-quality-pareto.svg",
        "fresh-vs-frozen.svg",
        "recovery-success.svg",
    ):
        assert name in script
