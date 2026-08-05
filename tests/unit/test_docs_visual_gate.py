"""The visual gate must fail the layouts v0.6.1 shipped and pass the new ones."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/fixtures/visual/legacy-metric-coverage-matrix.svg"
FIGURES = ROOT / "docs/alignment-lab"

#: Real rendered widths measured on the built site. The desktop content column
#: is 688 px at a 1440 px viewport; the mobile column is 358 px at 390 px.
DESKTOP_RENDERED_PX = 688.0
MOBILE_RENDERED_PX = 358.0


def _load(name: str, relative: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gate = _load("check_docs_visual", "scripts/check_docs_visual.py")
publisher = _load("publish_alignment_lab_artifacts", "scripts/publish_alignment_lab_artifacts.py")


# ------------------------------------------------------- generator contracts


def _payload() -> dict[str, Any]:
    return json.loads(
        (ROOT / "benchmarks/results/alignment-lab-v1.json").read_text(encoding="utf-8")
    )


def test_published_figures_are_desktop_and_mobile_pairs() -> None:
    figures = publisher.render_figures(_payload(), "0" * 64)
    assert set(figures) == {
        "delta-from-sft.svg",
        "delta-from-sft-mobile.svg",
        "outcome-cost-matrix.svg",
        "outcome-cost-matrix-mobile.svg",
    }
    for name, content in figures.items():
        assert (FIGURES / name).read_text(encoding="utf-8") == content
    assert not (FIGURES / "metric-coverage-matrix.svg").exists()


def test_mobile_figures_declare_a_narrow_canvas_and_readable_text() -> None:
    figures = publisher.render_figures(_payload(), "0" * 64)
    for name in ("delta-from-sft-mobile.svg", "outcome-cost-matrix-mobile.svg"):
        content = figures[name]
        assert f'width="{publisher.MOBILE_WIDTH}"' in content
        sizes = [float(value) for value in publisher._FONT_SIZE.findall(content)]
        assert min(sizes) >= publisher.MOBILE_MIN_FONT_PX


def test_a_scaled_desktop_canvas_is_rejected_as_a_mobile_figure() -> None:
    figures = publisher.render_figures(_payload(), "0" * 64)
    figures["delta-from-sft-mobile.svg"] = figures["delta-from-sft.svg"]
    with pytest.raises(ValueError, match="narrow canvas"):
        publisher.assert_chart_suitability(figures)


def test_metric_coverage_is_an_accessible_table_with_one_scope_statement() -> None:
    html = publisher.render_metric_coverage(_payload())
    assert publisher.COVERAGE_STATEMENT in html
    assert html.count("<tr>") == 1 + len(publisher.METHODS)
    assert html.count('scope="row"') == len(publisher.METHODS)
    assert html.count('data-label="Harmful compliance"') == len(publisher.METHODS)
    # One column-level statement, not six identical YES / NOT RUN cells.
    assert html.count("Sandbox endpoint measured:") == 1
    assert html.count("External safety benchmark executed:") == 1
    assert "YES" not in html
    assert "NOT RUN" not in html


def test_published_page_uses_responsive_pictures_with_correct_relative_paths() -> None:
    page = (FIGURES / "alignment-lab-v1.md").read_text(encoding="utf-8")
    assert page.count('<picture class="alignment-figure">') == 2
    # Raw HTML src values are not rewritten by MkDocs; the page lives one
    # directory below the figures once use_directory_urls builds it.
    assert page.count('srcset="../') == 2
    assert page.count('<img src="../') == 2
    assert 'media="(max-width: 900px)"' in page
    assert "metric-coverage-matrix.svg" not in page


# ------------------------------------------------------------ browser gate


def _browser():
    playwright = pytest.importorskip("playwright.sync_api")
    manager = playwright.sync_playwright()
    instance = manager.start()
    try:
        return manager, instance.chromium.launch()
    except Exception as exc:  # pragma: no cover - environment without browsers
        manager.__exit__(None, None, None)
        pytest.skip(f"chromium is not installed for Playwright: {exc}")


@pytest.fixture
def svg_page():
    manager, browser = _browser()
    page = browser.new_page()
    try:
        yield page
    finally:
        browser.close()
        manager.__exit__(None, None, None)


def test_the_v0_6_1_metric_coverage_headers_fail_the_new_gate(svg_page: Any) -> None:
    """The exact SVG v0.6.1 published must not survive the corrected gate."""
    assert FIXTURE.is_file()
    with pytest.raises(AssertionError) as excinfo:
        gate.assert_svg_document(
            svg_page,
            FIXTURE.resolve().as_uri(),
            rendered_width=DESKTOP_RENDERED_PX,
            min_font_px=gate.MIN_FONT_PX_WIDE,
        )
    message = str(excinfo.value)
    assert "overlapping text" in message
    assert "Sandbox endpoint" in message or "External safety" in message


def test_the_v0_6_1_layout_is_also_unreadable_at_a_phone_width(svg_page: Any) -> None:
    with pytest.raises(AssertionError) as excinfo:
        gate.assert_svg_document(
            svg_page,
            FIXTURE.resolve().as_uri(),
            rendered_width=MOBILE_RENDERED_PX,
            min_font_px=gate.MIN_FONT_PX_NARROW,
        )
    assert "below 11.0px" in str(excinfo.value) or "overlapping text" in str(excinfo.value)


@pytest.mark.parametrize(
    ("name", "width", "floor"),
    [
        ("delta-from-sft.svg", DESKTOP_RENDERED_PX, gate.MIN_FONT_PX_WIDE),
        ("outcome-cost-matrix.svg", DESKTOP_RENDERED_PX, gate.MIN_FONT_PX_WIDE),
        ("delta-from-sft-mobile.svg", MOBILE_RENDERED_PX, gate.MIN_FONT_PX_NARROW),
        ("outcome-cost-matrix-mobile.svg", MOBILE_RENDERED_PX, gate.MIN_FONT_PX_NARROW),
    ],
)
def test_the_published_figures_pass_the_new_gate(
    svg_page: Any, name: str, width: float, floor: float
) -> None:
    result = gate.assert_svg_document(
        svg_page,
        (FIGURES / name).resolve().as_uri(),
        rendered_width=width,
        min_font_px=floor,
    )
    assert result["textNodes"] > 0
    assert result["textOverlap"] == []
    assert result["occlusion"] == []
    assert result["tooSmall"] == []


def test_the_gate_inspects_untagged_text_nodes(svg_page: Any) -> None:
    """A collision between two untagged headers must be reported."""
    broken = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="400" height="120" '
        'viewBox="0 0 400 120"><title>t</title><desc>d</desc>'
        "<style>text{font-family:sans-serif;font-size:17px;fill:#fff}</style>"
        '<text x="20" y="60">Sandbox endpoint measured</text>'
        '<text x="120" y="60">External safety executed</text></svg>'
    )
    path = Path(gate.__file__).parent / "_gate_probe.svg"
    path.write_text(broken, encoding="utf-8")
    try:
        with pytest.raises(AssertionError, match="overlapping text"):
            gate.assert_svg_document(
                svg_page, path.resolve().as_uri(), rendered_width=400.0, min_font_px=10.5
            )
    finally:
        path.unlink(missing_ok=True)
