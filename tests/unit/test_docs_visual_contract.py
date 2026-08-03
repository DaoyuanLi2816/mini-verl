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
        "SVG label outside viewBox",
        "chart label overlap",
        "table is neither wrapped nor horizontally scrollable",
        "mobile bridge did not select the vertical layout",
        "chart label is smaller than",
    ):
        assert contract in script
