#!/usr/bin/env python3
"""Build-time browser assertions for documentation layout and generated SVGs.

The v0.6.1 gate measured every SVG standalone in a fixed 1400x1200 window and
derived its readability scale from a constant 820 px, so the 390 px page was
never actually measured and real mobile unreadability passed. It also compared
only ``[data-role]`` nodes, so two untagged ``<text>`` headers could overlap
undetected.

This gate instead measures what the browser actually rendered: for every image
that is *visible in the current viewport* it records the real bounding box,
re-renders that SVG at exactly that width, and asserts on the resulting
computed font sizes and glyph rectangles. ``currentSrc`` resolves ``<picture>``,
so a hidden desktop branch is never measured and never counted as passing.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import http.server
import json
import math
import re
import threading
from collections.abc import Iterator
from html import escape
from pathlib import Path
from typing import Any

VIEWPORTS = ((1440, 900), (1024, 768), (820, 1000), (390, 844))
PAGES = (
    "/direct-verl-config/",
    "/local-rl-workflow/",
    "/verl-compatibility-corpus/",
    "/",
    "/for-verl-users/",
    "/config-overrides/",
    "/immutable-plans/",
    "/profiles/",
    "/hardware-planning/",
    "/verl-opd-runtime/",
    "/verl-opd-reference-workload/",
    "/verl-opd-scaleout/",
    "/legacy-verl-bridge/",
    "/scaleout-materialization/",
    "/alignment-lab/alignment-lab-v1/",
    "/alignment-external/alignment-external-v1/",
    "/consumer-runtime/",
    "/recoverybench/recoverybench-v1/",
    "/verl-bridge/",
)

#: Minimum rendered font size for visible chart text, in real CSS pixels.
MIN_FONT_PX_NARROW = 11.0
MIN_FONT_PX_WIDE = 10.5
#: A viewport at or below this width is treated as a phone.
NARROW_VIEWPORT_PX = 480

#: No figure is exempt from the readability floor. v0.6.2 shipped four
#: pre-v0.6.2 figures on an explicit exemption list because they had no narrow
#: layout; v0.6.3 gave each of them a dedicated mobile SVG, so the list is
#: empty and the constant remains only to keep that intent asserted.
MOBILE_READABILITY_EXEMPT: frozenset[str] = frozenset()


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        del format, args


@contextlib.contextmanager
def _server(site: Path) -> Iterator[str]:
    handler = functools.partial(_QuietHandler, directory=str(site))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


# --------------------------------------------------------------- page checks

_VISIBLE_IMAGES_JS = """() => {
  const visible = (node) => {
    const style = getComputedStyle(node);
    if (style.display === 'none' || style.visibility === 'hidden') return false;
    if (parseFloat(style.opacity || '1') === 0) return false;
    const rect = node.getBoundingClientRect();
    return rect.width > 0 && rect.height > 0;
  };
  return [...document.querySelectorAll('img')]
    .filter(visible)
    .map((node) => {
      const rect = node.getBoundingClientRect();
      return {src: node.currentSrc || node.src, width: rect.width, height: rect.height,
              alt: node.getAttribute('alt') || '',
              // A 404 or unparsable figure still occupies an alt-text box, so
              // geometry alone cannot tell us the image actually loaded.
              loaded: node.complete && node.naturalWidth > 0};
    })
    .filter((item) => item.src.endsWith('.svg'));
}"""

_TABLE_JS = """(minFont) => {
  const problems = [];
  // Material's own table scrollwrap deliberately bleeds into .md-typeset's
  // horizontal padding, so .md-typeset -- not .md-content__inner -- is the
  // box a table may legitimately fill.
  const typeset = document.querySelector('.md-typeset');
  const columnRight = typeset ? typeset.getBoundingClientRect().right : Infinity;
  for (const table of document.querySelectorAll('.md-typeset table')) {
    const style = getComputedStyle(table);
    if (style.display === 'none') continue;
    const scroller = table.closest('.md-typeset__scrollwrap, .coverage-scroll');
    const scrolls = scroller && ['auto', 'scroll'].includes(getComputedStyle(scroller).overflowX);
    if (table.scrollWidth > table.clientWidth + 1 && !scrolls) {
      problems.push(`table overflows without a scroll container: ${table.textContent.slice(0, 60)}`);
    }
    const box = (scroller || table).getBoundingClientRect();
    if (box.right > columnRight + 1) {
      problems.push(`table exceeds the content column by ${(box.right - columnRight).toFixed(1)}px`);
    }
    for (const cell of table.querySelectorAll('th, td')) {
      const cellStyle = getComputedStyle(cell);
      if (cellStyle.display === 'none' || !cell.textContent.trim()) continue;
      const size = parseFloat(cellStyle.fontSize || '0');
      if (size > 0 && size < minFont) {
        problems.push(`table cell renders at ${size.toFixed(1)}px: ${cell.textContent.trim().slice(0, 40)}`);
      }
      const label = getComputedStyle(cell, '::before');
      if (label.content && label.content !== 'none') {
        const labelSize = parseFloat(label.fontSize || '0');
        if (labelSize > 0 && labelSize < minFont) {
          problems.push(`card label renders at ${labelSize.toFixed(1)}px`);
        }
      }
    }
  }
  return problems;
}"""

_CARD_LABELS_JS = """() => {
  const rows = [...document.querySelectorAll('.coverage-table tbody tr')];
  if (!rows.length) return null;
  const cells = rows[0].querySelectorAll('td');
  if (!cells.length) return 'coverage rows have no data cells';
  for (const cell of cells) {
    const label = getComputedStyle(cell, '::before').content;
    if (!label || label === 'none' || label === '""') {
      return `coverage card cell has no visible column label: ${cell.textContent.trim().slice(0, 30)}`;
    }
  }
  return null;
}"""


def _assert_page(page: Any, *, route: str, width: int) -> list[dict[str, Any]]:
    """Assert page-level layout and return every *visible* SVG with its real box."""
    overflow = page.evaluate(
        "() => ({client: document.documentElement.clientWidth,"
        " scroll: document.documentElement.scrollWidth})"
    )
    if overflow["scroll"] > overflow["client"] + 1:
        raise AssertionError(f"horizontal document overflow at {route}: {overflow}")

    figure_problems = page.evaluate(
        """() => {
          const content = document.querySelector('.md-content__inner');
          if (!content) return ['missing content column'];
          const cr = content.getBoundingClientRect();
          return [...document.querySelectorAll('.md-typeset img, .md-typeset picture')]
            .map((node) => ({node, rect: node.getBoundingClientRect()}))
            .filter(({rect}) => rect.width > cr.width + 1 || rect.left < cr.left - 1 || rect.right > cr.right + 1)
            .map(({node, rect}) => `${node.tagName}:${node.getAttribute('src') || ''}:${rect.width}/${cr.width}`);
        }"""
    )
    if figure_problems:
        raise AssertionError(f"figure exceeds content column at {route}: {figure_problems}")

    minimum = MIN_FONT_PX_NARROW if width <= NARROW_VIEWPORT_PX else MIN_FONT_PX_WIDE
    table_problems = page.evaluate(_TABLE_JS, minimum)
    if table_problems:
        raise AssertionError(f"responsive table problem at {route} @{width}px: {table_problems}")

    if route == "/alignment-lab/alignment-lab-v1/" and width <= NARROW_VIEWPORT_PX:
        card_problem = page.evaluate(_CARD_LABELS_JS)
        if card_problem:
            raise AssertionError(f"metric-coverage card layout at {width}px: {card_problem}")

    images = page.evaluate(_VISIBLE_IMAGES_JS)
    for image in images:
        if not image["alt"].strip():
            raise AssertionError(f"figure without alt text at {route}: {image['src']}")
        if not image["loaded"]:
            raise AssertionError(
                f"figure did not load at {route} @{width}px: {image['src']} "
                "(raw HTML src values are not rewritten by MkDocs; check the relative path)"
            )

    if route == "/verl-bridge/" and width == 390:
        current = page.locator("picture.bridge-architecture img").evaluate("img => img.currentSrc")
        if not current.endswith("verl-bridge-architecture-mobile.svg"):
            raise AssertionError(f"mobile bridge did not select the vertical layout: {current}")
    if route == "/alignment-lab/alignment-lab-v1/" and width == 390:
        selected = page.locator("picture.alignment-figure img").evaluate_all(
            "nodes => nodes.map(node => node.currentSrc)"
        )
        if not selected or not all(src.endswith("-mobile.svg") for src in selected):
            raise AssertionError(f"mobile alignment figures did not activate: {selected}")
    if route == "/alignment-external/alignment-external-v1/" and width == 390:
        selected = page.locator("picture.alignment-figure img").evaluate_all(
            "nodes => nodes.map(node => node.currentSrc)"
        )
        if len(selected) != 2 or not all(src.endswith("-mobile.svg") for src in selected):
            raise AssertionError(f"mobile external-alignment figures did not activate: {selected}")

    return images


# ---------------------------------------------------------------- SVG checks

_SVG_JS = """(options) => {
  const svg = document.documentElement;
  if (!svg || svg.tagName.toLowerCase() !== 'svg' || !svg.viewBox || !svg.viewBox.baseVal) {
    return {error: 'missing SVG viewBox'};
  }
  const vb = svg.viewBox.baseVal;
  if (!vb.width || !vb.height) return {error: 'empty SVG viewBox'};
  // Reproduce the exact geometry the page rendered: width is what the browser
  // gave the <img>, height follows from the viewBox aspect ratio.
  svg.setAttribute('width', String(options.width));
  svg.setAttribute('height', String((vb.height * options.width) / vb.width));
  svg.style.display = 'block';

  const visible = (node) => {
    const style = getComputedStyle(node);
    if (style.display === 'none' || style.visibility === 'hidden') return false;
    if (parseFloat(style.opacity || '1') === 0) return false;
    const rect = node.getBoundingClientRect();
    return rect.width > 0.5 && rect.height > 0.5;
  };
  const root = svg.getBoundingClientRect();

  // EVERY visible text node, not only the ones an author remembered to tag.
  const texts = [...svg.querySelectorAll('text')]
    .filter((node) => (node.textContent || '').trim() && visible(node))
    .map((node) => ({
      node,
      rect: node.getBoundingClientRect(),
      size: parseFloat(getComputedStyle(node).fontSize || '0') *
            Math.hypot(node.getScreenCTM().c, node.getScreenCTM().d),
      role: node.getAttribute('data-role') || '',
      text: (node.textContent || '').trim(),
    }));

  const outside = [];
  for (const {node, rect, text} of texts) {
    if (rect.left < root.left - 1 || rect.top < root.top - 1 ||
        rect.right > root.right + 1 || rect.bottom > root.bottom + 1) {
      outside.push(`${node.tagName}:${text.slice(0, 60)}`);
    }
  }

  const area = (a, b) => Math.max(0, Math.min(a.right, b.right) - Math.max(a.left, b.left)) *
                         Math.max(0, Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top));
  const intersects = (a, b, slack) =>
    Math.min(a.right, b.right) - Math.max(a.left, b.left) > slack &&
    Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top) > slack;

  const textOverlap = [];
  for (let i = 0; i < texts.length; i += 1) {
    for (let j = i + 1; j < texts.length; j += 1) {
      if (intersects(texts[i].rect, texts[j].rect, options.tolerance)) {
        textOverlap.push(`"${texts[i].text.slice(0, 40)}" x "${texts[j].text.slice(0, 40)}"`);
      }
    }
  }

  // Label-to-mark occlusion: a printed value must never sit on top of a mark.
  const marks = [...svg.querySelectorAll('[data-encoding="seed-point"], [data-value]')]
    .filter((node) => node.tagName.toLowerCase() !== 'text' && visible(node))
    .map((node) => ({node, rect: node.getBoundingClientRect()}));
  // Tighter slack than text-to-text: a printed value touching its own mark is
  // already a readability defect, not a near miss.
  const occlusion = [];
  for (const {rect, text} of texts) {
    for (const mark of marks) {
      if (intersects(rect, mark.rect, options.markTolerance)) {
        occlusion.push(`"${text.slice(0, 40)}" over <${mark.node.tagName}>`);
      }
    }
  }

  // Check actual outlines, not their bounding boxes: a routed elbow can safely
  // surround a label, but a connector, card edge or untagged marker cannot cut it.
  // Zero-width vertical lines are visible too, unlike image/text boxes above.
  const graphics = [...svg.querySelectorAll('path,line,polyline,polygon,rect,circle,ellipse')]
    .filter(node => {
      if (node.closest('defs,clipPath,mask')) return false;
      const style = getComputedStyle(node);
      if (style.display === 'none' || style.visibility === 'hidden') return false;
      if (parseFloat(style.opacity || '1') === 0) return false;
      return true;
    });
  const graphicOverlap = [];
  const contains = (a, b, pad = 0) => a.left <= b.left - pad && a.top <= b.top - pad &&
    a.right >= b.right + pad && a.bottom >= b.bottom + pad;
  for (const {node: textNode, rect, text} of texts) {
    for (const node of graphics) {
      const r = node.getBoundingClientRect();
      if (r.right < rect.left || r.left > rect.right ||
          r.bottom < rect.top || r.top > rect.bottom) continue;
      const earlier = !!(node.compareDocumentPosition(textNode) & Node.DOCUMENT_POSITION_FOLLOWING);
      if (earlier && node.tagName === 'rect' && contains(r, rect, 1)) continue;
      const style = getComputedStyle(node);
      // Background washes may sit behind text, never paint over it afterwards.
      if (earlier && style.stroke === 'none' && node.tagName === 'path' &&
          r.width * r.height >= root.width * root.height * .05) continue;
      const matrix = node.getScreenCTM();
      const center = new DOMPoint((rect.left + rect.right) / 2, (rect.top + rect.bottom) / 2);
      if (!earlier && style.fill !== 'none' && +style.fillOpacity > 0 &&
          node.isPointInFill(center.matrixTransform(matrix.inverse()))) {
        graphicOverlap.push(`"${text.slice(0, 55)}" covered by <${node.tagName}>`);
        continue;
      }
      const length = node.getTotalLength();
      const step = Math.max(.25, 1 / Math.max(Math.hypot(matrix.a, matrix.b), .01));
      for (let at = 0; at <= length; at += step) {
        const p = node.getPointAtLength(at).matrixTransform(matrix);
        if (p.x <= rect.left + 1 || p.x >= rect.right - 1 ||
            p.y <= rect.top + 1 || p.y >= rect.bottom - 1) continue;
        // A solid label backing can legitimately mask a grid behind it.
        const masked = graphics.some(cover => {
          if (cover === node || cover.tagName !== 'rect') return false;
          if (!(node.compareDocumentPosition(cover) & Node.DOCUMENT_POSITION_FOLLOWING)) return false;
          const style = getComputedStyle(cover);
          if (style.fill === 'none' || +style.fillOpacity < 1 || +style.opacity < 1) return false;
          return cover.isPointInFill(new DOMPoint(p.x, p.y).matrixTransform(cover.getScreenCTM().inverse()));
        });
        if (!masked) {
          graphicOverlap.push(`"${text.slice(0, 55)}" x <${node.tagName}>`);
          break;
        }
      }
    }
  }

  const legends = [...svg.querySelectorAll('[data-role="legend"]')].filter(visible);
  const plots = [...svg.querySelectorAll('[data-role="plot-region"]')].filter(visible);
  const legendPlotOverlap = [];
  for (const legend of legends) {
    for (const plot of plots) {
      if (area(legend.getBoundingClientRect(), plot.getBoundingClientRect()) > 1) {
        legendPlotOverlap.push('legend overlaps plotting region');
      }
    }
  }

  const tooSmall = texts
    .filter((item) => item.size > 0 && item.size < options.minFont)
    .map((item) => `${item.size.toFixed(2)}px:${item.text.slice(0, 50)}`);

  return {outside, textOverlap: [...new Set(textOverlap)], occlusion: [...new Set(occlusion)],
          graphicOverlap: [...new Set(graphicOverlap)],
          legendPlotOverlap, tooSmall, textNodes: texts.length,
          renderedWidth: root.width, renderedHeight: root.height};
}"""


def assert_svg_document(
    page: Any,
    url: str,
    *,
    rendered_width: float,
    min_font_px: float,
    enforce_font: bool = True,
    tolerance: float = 1.5,
    mark_tolerance: float = 0.25,
) -> dict[str, Any]:
    """Assert one SVG at the exact width the page rendered it.

    Raises :class:`AssertionError` on out-of-bounds text, any visible
    text-to-text overlap, label-to-mark occlusion, legend/plot overlap, or text
    below ``min_font_px`` when ``enforce_font`` is set.
    """
    width = max(1, math.floor(rendered_width))
    page.set_viewport_size({"width": width, "height": 1200})
    page.goto(url, wait_until="load")
    page.evaluate("document.fonts.ready")
    result = page.evaluate(
        _SVG_JS,
        {
            "width": width,
            "minFont": min_font_px,
            "tolerance": tolerance,
            "markTolerance": mark_tolerance,
        },
    )
    if result.get("error"):
        raise AssertionError(f"{url}: {result['error']}")
    where = f"{url} rendered at {width}px"
    if not result["textNodes"]:
        raise AssertionError(f"{where}: no visible text found; the figure cannot be read")
    if result["outside"]:
        raise AssertionError(f"{where}: text outside the viewBox: {result['outside']}")
    if result["textOverlap"]:
        raise AssertionError(f"{where}: overlapping text: {result['textOverlap']}")
    if result["occlusion"]:
        raise AssertionError(f"{where}: label occludes a data mark: {result['occlusion']}")
    if result["graphicOverlap"]:
        raise AssertionError(f"{where}: graphic intersects text: {result['graphicOverlap']}")
    if result["legendPlotOverlap"]:
        raise AssertionError(f"{where}: legend overlaps the plotting region")
    if enforce_font and result["tooSmall"]:
        raise AssertionError(f"{where}: text below {min_font_px}px: {result['tooSmall']}")
    return result


# ------------------------------------------------------------------- driver


def readme_figure_preview(root: Path, base_url: str, language: str = "README.md") -> str:
    """Render the actual README picture markup in a GitHub-width content column.

    Badge/network requests and Markdown prose do not determine SVG geometry.
    Keep picture sources intact so this also catches missing mobile branches.
    """
    source = (root / language).read_text(encoding="utf-8")
    pictures = re.findall(r"<picture\b[^>]*>.*?</picture>|<img\b[^>]*>", source, re.S)
    figures = "\n".join(pictures)
    figures = re.sub(
        r"https://raw\.githubusercontent\.com/DaoyuanLi2816/mini-verl/[^/]+/",
        f"{base_url}/",
        figures,
    )
    return (
        f'<html><head><base href="{base_url}/"><style>'
        "body{margin:0;background:#0d1117;color:#e6edf3;font:16px Arial}"
        ".md-content__inner{max-width:880px;margin:0 auto;padding:16px}"
        "picture{display:block;margin:0 0 32px}img{max-width:100%;height:auto}"
        "</style></head><body><main class='md-content__inner md-typeset'>"
        f"<h1>{escape(language)} figures</h1>{figures}</main></body></html>"
    )


def check(site: Path, screenshots: Path) -> None:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:  # pragma: no cover - exercised in CI setup
        raise SystemExit("install the pinned Playwright dev dependency first") from exc

    screenshots.mkdir(parents=True, exist_ok=True)
    # One entry per (svg, viewport): the same file is checked again whenever a
    # different viewport renders it at a different real width.
    measured: dict[tuple[str, int], float] = {}
    root = Path(__file__).resolve().parents[1]
    with _server(site) as base_url, _server(root) as repo_url, sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            for width, height in VIEWPORTS:
                context = browser.new_context(
                    viewport={"width": width, "height": height},
                    color_scheme="dark",
                    device_scale_factor=1,
                    locale="en-US",
                    reduced_motion="reduce",
                )
                page = context.new_page()
                for route in PAGES:
                    page.goto(f"{base_url}{route}", wait_until="networkidle")
                    for image in _assert_page(page, route=route, width=width):
                        key = (image["src"], width)
                        measured[key] = max(measured.get(key, 0.0), image["width"])
                    slug = "home" if route == "/" else re.sub(r"[^a-z0-9]+", "-", route).strip("-")
                    page.screenshot(
                        path=str(screenshots / f"{width}x{height}-{slug}.png"),
                        full_page=True,
                        animations="disabled",
                    )
                for language in ("README.md", "README.zh-CN.md"):
                    page.set_content(readme_figure_preview(root, repo_url, language))
                    page.wait_for_function(
                        "[...document.images].every(img => img.complete && img.naturalWidth > 0)"
                    )
                    for image in _assert_page(page, route=language, width=width):
                        measured[(image["src"], width)] = image["width"]
                    if width <= NARROW_VIEWPORT_PX:
                        selected = page.locator("picture img").evaluate_all(
                            "nodes => nodes.map(node => node.currentSrc)"
                        )
                        if not all(src.endswith("-mobile.svg") for src in selected):
                            raise AssertionError(f"README mobile layout missing: {selected}")
                    page.screenshot(
                        path=str(screenshots / f"{width}x{height}-{language}.png"),
                        full_page=True,
                        animations="disabled",
                    )
                context.close()

            svg_page = browser.new_page()
            # Include unembedded/social/legacy assets too. Native-size geometry
            # is independent of whether MkDocs currently links a particular SVG.
            for asset in sorted((root / "docs").rglob("*.svg")):
                url = f"{repo_url}/{asset.relative_to(root).as_posix()}"
                svg_page.goto(url)
                native = svg_page.evaluate("document.documentElement.viewBox.baseVal.width")
                assert_svg_document(svg_page, url, rendered_width=native, min_font_px=0)
                slug = asset.relative_to(root / "docs").as_posix().replace("/", "-")
                svg_page.locator("svg").screenshot(path=str(screenshots / f"native-{slug}.png"))
            for (url, viewport), rendered in sorted(measured.items()):
                narrow = viewport <= NARROW_VIEWPORT_PX
                minimum = MIN_FONT_PX_NARROW if narrow else MIN_FONT_PX_WIDE
                # Every figure is enforced at every viewport. v0.6.2 carried four
                # pre-v0.6.2 figures on an exemption list; v0.6.3 gave each of
                # them a real narrow layout and the list is gone.
                assert_svg_document(
                    svg_page,
                    url,
                    rendered_width=rendered,
                    min_font_px=minimum,
                    enforce_font=True,
                )
            svg_page.close()
        finally:
            browser.close()

    print(f"checked {len(measured)} rendered SVG instances across {len(VIEWPORTS)} viewports")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, default=Path("site"))
    parser.add_argument("--screenshots", type=Path, default=Path("docs-visual-screenshots"))
    parser.add_argument(
        "--report", type=Path, default=None, help="Optional JSON summary of what was measured."
    )
    args = parser.parse_args()
    if not args.site.is_dir():
        parser.error(f"site directory does not exist: {args.site}")
    check(args.site.resolve(), args.screenshots.resolve())
    if args.report:
        args.report.write_text(
            json.dumps(
                {
                    "viewports": [list(item) for item in VIEWPORTS],
                    "pages": list(PAGES),
                    "min_font_px_narrow": MIN_FONT_PX_NARROW,
                    "min_font_px_wide": MIN_FONT_PX_WIDE,
                    "mobile_readability_exempt": sorted(MOBILE_READABILITY_EXEMPT),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
