#!/usr/bin/env python3
"""Build-time browser assertions for documentation layout and generated SVGs."""

from __future__ import annotations

import argparse
import contextlib
import functools
import http.server
import re
import threading
from pathlib import Path
from typing import Any

VIEWPORTS = ((1440, 900), (1024, 768), (820, 1000), (390, 844))
PAGES = (
    "/",
    "/alignment-lab/alignment-lab-v1/",
    "/consumer-runtime/",
    "/recoverybench/recoverybench-v1/",
    "/verl-bridge/",
)


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        del format, args


@contextlib.contextmanager
def _server(site: Path):
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


def _assert_page(page: Any, *, route: str, width: int) -> list[str]:
    overflow = page.evaluate(
        "() => ({client: document.documentElement.clientWidth, scroll: document.documentElement.scrollWidth})"
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

    table_problems = page.evaluate(
        """() => [...document.querySelectorAll('.md-typeset table')].filter((table) => {
          if (table.scrollWidth <= table.clientWidth + 1) return false;
          const wrapper = table.closest('.md-typeset__scrollwrap');
          if (!wrapper) return true;
          const overflow = getComputedStyle(wrapper).overflowX;
          return !['auto', 'scroll'].includes(overflow);
        }).map((table) => table.textContent.slice(0, 80))"""
    )
    if table_problems:
        raise AssertionError(
            f"table is neither wrapped nor horizontally scrollable at {route}: {table_problems}"
        )

    if route == "/verl-bridge/" and width == 390:
        current = page.locator("picture.bridge-architecture img").evaluate("img => img.currentSrc")
        if not current.endswith("verl-bridge-architecture-mobile.svg"):
            raise AssertionError(f"mobile bridge did not select the vertical layout: {current}")

    return page.locator('img[src*=".svg"], picture img').evaluate_all(
        "nodes => nodes.map(node => node.currentSrc || node.src).filter(src => src.endsWith('.svg'))"
    )


def _assert_svg(page: Any, url: str) -> None:
    page.set_viewport_size({"width": 1400, "height": 1200})
    page.goto(url, wait_until="load")
    result = page.evaluate(
        """() => {
          const svg = document.querySelector('svg');
          if (!svg || !svg.viewBox || !svg.viewBox.baseVal) return {error: 'missing SVG viewBox'};
          const root = svg.getBoundingClientRect();
          const vb = svg.viewBox.baseVal;
          const scale = Math.min(820, vb.width) / vb.width;
          const outside = [];
          for (const node of svg.querySelectorAll('text, [data-role]')) {
            const rect = node.getBoundingClientRect();
            if (rect.left < root.left - 1 || rect.top < root.top - 1 ||
                rect.right > root.right + 1 || rect.bottom > root.bottom + 1) {
              outside.push(`${node.tagName}:${(node.textContent || '').trim().slice(0, 60)}`);
            }
          }
          const labels = [...svg.querySelectorAll('[data-role="chart-label"], [data-role="diagram-label"]')];
          const overlap = [];
          for (let i = 0; i < labels.length; i += 1) {
            const a = labels[i].getBoundingClientRect();
            for (let j = i + 1; j < labels.length; j += 1) {
              const b = labels[j].getBoundingClientRect();
              const area = Math.max(0, Math.min(a.right, b.right) - Math.max(a.left, b.left)) *
                Math.max(0, Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top));
              if (area > 1) overlap.push(`${i}:${j}`);
            }
          }
          const legends = [...svg.querySelectorAll('[data-role="legend"]')];
          const plots = [...svg.querySelectorAll('[data-role="plot-region"]')];
          const legendPlotOverlap = [];
          for (const legend of legends) for (const plot of plots) {
            const a = legend.getBoundingClientRect(); const b = plot.getBoundingClientRect();
            if (Math.max(0, Math.min(a.right, b.right) - Math.max(a.left, b.left)) *
                Math.max(0, Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top)) > 1) {
              legendPlotOverlap.push('legend overlaps plotting region');
            }
          }
          const tooSmall = labels.filter((node) => {
            const size = parseFloat(getComputedStyle(node).fontSize || '0') * scale;
            return size > 0 && size < 10.5;
          }).map((node) => `${getComputedStyle(node).fontSize}:${(node.textContent || '').trim().slice(0, 50)}`);
          return {outside, overlap, legendPlotOverlap, tooSmall};
        }"""
    )
    if result.get("error"):
        raise AssertionError(result["error"])
    if result["outside"]:
        raise AssertionError(f"SVG label outside viewBox in {url}: {result['outside']}")
    if result["overlap"]:
        raise AssertionError(f"chart label overlap in {url}: {result['overlap']}")
    if result["legendPlotOverlap"]:
        raise AssertionError(f"legend overlaps the plotting region in {url}")
    if result["tooSmall"]:
        raise AssertionError(
            f"chart label is smaller than 10.5 px at 820 px in {url}: {result['tooSmall']}"
        )


def check(site: Path, screenshots: Path) -> None:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:  # pragma: no cover - exercised in CI setup
        raise SystemExit("install the pinned Playwright dev dependency first") from exc

    screenshots.mkdir(parents=True, exist_ok=True)
    with _server(site) as base_url, sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        svg_urls: set[str] = set()
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
                    svg_urls.update(_assert_page(page, route=route, width=width))
                    slug = "home" if route == "/" else re.sub(r"[^a-z0-9]+", "-", route).strip("-")
                    page.screenshot(
                        path=str(screenshots / f"{width}x{height}-{slug}.png"),
                        full_page=True,
                        animations="disabled",
                    )
                context.close()
            svg_page = browser.new_page()
            for url in sorted(svg_urls):
                _assert_svg(svg_page, url)
            svg_page.close()
        finally:
            browser.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, default=Path("site"))
    parser.add_argument("--screenshots", type=Path, default=Path("docs-visual-screenshots"))
    args = parser.parse_args()
    if not args.site.is_dir():
        parser.error(f"site directory does not exist: {args.site}")
    check(args.site.resolve(), args.screenshots.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
