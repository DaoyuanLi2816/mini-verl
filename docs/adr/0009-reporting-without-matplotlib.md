# 0009. Reporting with hand-rolled inline SVG, no plotting library

Status: Accepted, 2026-07-27.

## Context

`miniverl report <run-dir>` renders an HTML report of a training run: metric
curves, span-type breakdowns, a per-token divergence strip and the run's
provenance. Two constraints shape how those charts are produced.

First, **the report must render offline from the base install**. A run
directory is produced on a GPU machine and read somewhere else -- a laptop, a
CI artifact, an issue attachment. `src/miniverl/reporting/data.py` is
deliberately torch-free: it reads only JSON, JSONL and YAML, so a run directory
copied off the GPU box can be reported on by a machine that has never had
torch installed.

Second, **the output must be a single file with no network access**. A report
that fetches a chart library from a CDN is blank in an air-gapped environment,
blank when the CDN changes a URL, and a privacy question in any environment.

Those two constraints together rule out both matplotlib (a large required
dependency, pulling in numpy, that would have to be imported to produce a PNG)
and any JavaScript charting library (external requests, or an inlined bundle).

## Decision

Charts are hand-rolled inline SVG. `src/miniverl/reporting/charts.py` provides
four primitives and nothing else:

- `line_chart` -- multi-series `(name, xs, ys)` with gridlines, tick labels and
  a legend, degrading to a single point when a series has one sample;
- `bar_chart` -- horizontal categorical bars;
- `sparkline` -- a small trend line for inline use;
- `token_strip` -- the per-token divergence view, where each cell is one
  stored, model-generated token, colour intensity is that token's divergence,
  and the tooltip carries the span type, weight, teacher entropy and the two
  argmax tokens.

The `PALETTE` is five colours chosen to stay legible in both light and dark
themes, and the template carries a `prefers-color-scheme: dark` block. All text
passed into an SVG goes through `html.escape`, and charts carry
`role="img"` with an `aria-label`.

`token_strip`'s docstring states the rule that keeps the report honest:
"Nothing hidden or unstored is rendered." Cells come from records that exist in
the run's artifacts. Where a value is a coarse-grained lower bound rather than
an exact quantity -- teacher entropy from `bucketed_teacher_entropy`, see
ADR 0004 -- the surrounding text labels it as such.

**There is no `reporting` extra in `pyproject.toml`.** The extras are `train`,
`cuda` and `dev`. `jinja2` is a base dependency, used for one template,
`src/miniverl/reporting/templates/report.html.j2`, which inlines its CSS in a
`<style>` block and contains no `<script>`, no `<link rel>` and no external
URL. The rendered page states this at the bottom: "This report is
self-contained: no scripts, no external requests, no fonts to fetch."

The same `ReportData` feeds three renderers -- HTML, Markdown
(`summary.md`, for pasting into an issue) and a JSON summary -- so the three
cannot disagree.

## Consequences

Positive:

- The report opens from `file://` with no network, no fonts to fetch and no
  scripts to run.
- No plotting dependency enters the base install, and no plotting dependency
  can break the report by changing its API.
- The rendered SVG is text, so a report diffs meaningfully in version control
  and can be inspected without a browser.
- Chart code is testable as string output;
  `tests/unit/test_inspection_reporting.py` exercises it without a display or a
  headless browser.

Negative:

- The chart vocabulary is fixed at four primitives. Anything else -- a
  histogram, a scatter plot, a stacked area -- is new code, not a new library
  call.
- The charts are static. There is no zoom, no pan, no hover crosshair; the only
  interactivity is the browser's native `title` tooltip.
- Axis handling is minimal: five gridlines, linear scale only, no log axis, no
  date formatting, no automatic tick rounding beyond `_fmt`. Degenerate ranges
  are handled by nudging the bound rather than by a real nice-numbers
  algorithm.
- The layout is hand-computed in pixels, so a long series name or a large token
  strip can crowd the plot area. `token_strip` caps at `max_tokens` (400 by
  default, `report.max_tokens_per_trajectory` in the config).
- Nobody is maintaining these primitives except this project.

## Alternatives considered

**matplotlib, rendering PNGs embedded as data URIs.** Rejected: it would be
either a large required dependency or an optional one whose absence silently
degrades the report, and it pulls numpy into the report path, which is what
`data.py` avoids.

**A JavaScript charting library from a CDN.** Rejected outright: the report
would not render offline.

**The same library inlined into the HTML.** Rejected: it makes every report
file hundreds of kilobytes of vendored third-party code, adds a
licence-attribution obligation per report, and the report is not interactive
enough to earn it.

**No charts, tables only.** Rejected: the token-level divergence strip is the
part of the report that shows *where* the loss is, and a table does not convey
it.
