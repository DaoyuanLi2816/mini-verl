from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


def _load() -> Any:
    spec = importlib.util.spec_from_file_location(
        "check_docs_visual", ROOT / "scripts/check_docs_visual.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_no_figure_is_exempt_from_the_mobile_readability_floor() -> None:
    """v0.6.2 carried four figures on an exemption list; v0.6.3 must carry none."""
    gate = _load()
    assert frozenset() == gate.MOBILE_READABILITY_EXEMPT
    script = (ROOT / "scripts/check_docs_visual.py").read_text(encoding="utf-8")
    # The floor is applied unconditionally, not through a per-figure opt-out.
    assert "enforce_font=True" in script


def test_every_legacy_figure_now_has_a_narrow_layout() -> None:
    """Each formerly exempt figure ships a real mobile SVG and is referenced."""
    pairs = {
        "docs/consumer-runtime-v1-pareto.svg": "docs/consumer-runtime-v1-pareto-mobile.svg",
        "docs/recoverybench/cost-quality-pareto.svg": (
            "docs/recoverybench/cost-quality-pareto-mobile.svg"
        ),
        "docs/recoverybench/fresh-vs-frozen.svg": ("docs/recoverybench/fresh-vs-frozen-mobile.svg"),
        "docs/recoverybench/recovery-success.svg": (
            "docs/recoverybench/recovery-success-mobile.svg"
        ),
    }
    pages = "\n".join(path.read_text(encoding="utf-8") for path in (ROOT / "docs").rglob("*.md"))
    for desktop, mobile in pairs.items():
        mobile_path = ROOT / mobile
        assert mobile_path.is_file(), f"{mobile} is missing"
        content = mobile_path.read_text(encoding="utf-8")
        # A narrow canvas, not the 1120 px desktop chart scaled down.
        assert 'width="390"' in content
        sizes = [float(value) for value in re.findall(r"font-size:([0-9.]+)px", content)]
        assert sizes and min(sizes) >= 14.0
        # Selected by a media query, so the desktop source stays inactive.
        assert f'srcset="{Path(mobile).name}"' in pages or f"/{Path(mobile).name}" in pages
        assert Path(desktop).name in pages
