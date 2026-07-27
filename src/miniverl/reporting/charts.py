"""Hand-rolled inline SVG charts.

No matplotlib, no plotly, no CDN.  Reports must render offline from a bare
``pip install miniverl``, and a chart library would either add a heavy required
dependency or make the report depend on the network.  These four primitives
cover everything the report shows.
"""

from __future__ import annotations

import html
from collections.abc import Sequence

__all__ = ["line_chart", "bar_chart", "token_strip", "sparkline", "PALETTE"]

#: Colours chosen to stay legible in both light and dark themes.
PALETTE: tuple[str, ...] = ("#2f7ed8", "#e5734f", "#3f9c66", "#8c6bb1", "#c0392b")


def _fmt(value: float) -> str:
    if value == 0:
        return "0"
    if abs(value) >= 1000 or abs(value) < 0.01:
        return f"{value:.2e}"
    return f"{value:.3g}"


def line_chart(
    series: Sequence[tuple[str, Sequence[float], Sequence[float]]],
    *,
    width: int = 720,
    height: int = 240,
    x_label: str = "",
    y_label: str = "",
    title: str = "",
) -> str:
    """Multi-series line chart. Each series is ``(name, xs, ys)``."""
    usable = [(n, list(xs), list(ys)) for n, xs, ys in series if len(xs) >= 1 and len(xs) == len(ys)]
    if not usable:
        return '<p class="muted">no data</p>'
    pad_l, pad_r, pad_t, pad_b = 56, 110, 24, 36
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    all_x = [x for _, xs, _ in usable for x in xs]
    all_y = [y for _, _, ys in usable for y in ys]
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    if x_max == x_min:
        x_max = x_min + 1
    if y_max == y_min:
        y_max = y_min + (abs(y_min) or 1) * 0.1

    def px(x: float) -> float:
        return pad_l + (x - x_min) / (x_max - x_min) * plot_w

    def py(y: float) -> float:
        return pad_t + plot_h - (y - y_min) / (y_max - y_min) * plot_h

    parts = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{html.escape(title or y_label)}" preserveAspectRatio="xMidYMid meet">'
    ]
    for i in range(5):
        y = y_min + (y_max - y_min) * i / 4
        yy = py(y)
        parts.append(
            f'<line class="grid" x1="{pad_l}" y1="{yy:.1f}" x2="{pad_l + plot_w}" y2="{yy:.1f}"/>'
        )
        parts.append(
            f'<text class="tick" x="{pad_l - 6}" y="{yy + 4:.1f}" text-anchor="end">{_fmt(y)}</text>'
        )
    parts.append(
        f'<line class="axis" x1="{pad_l}" y1="{pad_t + plot_h}" '
        f'x2="{pad_l + plot_w}" y2="{pad_t + plot_h}"/>'
    )
    for i, (name, xs, ys) in enumerate(usable):
        colour = PALETTE[i % len(PALETTE)]
        points = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in zip(xs, ys))
        if len(xs) == 1:
            parts.append(
                f'<circle cx="{px(xs[0]):.1f}" cy="{py(ys[0]):.1f}" r="3.5" fill="{colour}"/>'
            )
        else:
            parts.append(
                f'<polyline fill="none" stroke="{colour}" stroke-width="2" points="{points}"/>'
            )
        legend_y = pad_t + 14 + i * 16
        parts.append(
            f'<rect x="{pad_l + plot_w + 12}" y="{legend_y - 8}" width="10" height="10" '
            f'fill="{colour}"/>'
        )
        parts.append(
            f'<text class="legend" x="{pad_l + plot_w + 27}" y="{legend_y + 1}">'
            f"{html.escape(name)}</text>"
        )
    parts.append(
        f'<text class="tick" x="{pad_l + plot_w / 2:.0f}" y="{height - 8}" '
        f'text-anchor="middle">{html.escape(x_label)}</text>'
    )
    if y_label:
        parts.append(
            f'<text class="tick" x="12" y="{pad_t + plot_h / 2:.0f}" '
            f'transform="rotate(-90 12 {pad_t + plot_h / 2:.0f})" text-anchor="middle">'
            f"{html.escape(y_label)}</text>"
        )
    parts.append("</svg>")
    return "".join(parts)


def bar_chart(
    items: Sequence[tuple[str, float]],
    *,
    width: int = 720,
    bar_height: int = 22,
    title: str = "",
) -> str:
    """Horizontal bar chart for categorical counts."""
    data = [(str(k), float(v)) for k, v in items if v is not None]
    if not data:
        return '<p class="muted">no data</p>'
    label_w = 190
    max_value = max(v for _, v in data) or 1.0
    height = len(data) * (bar_height + 6) + 12
    plot_w = width - label_w - 70
    parts = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{html.escape(title)}" preserveAspectRatio="xMidYMid meet">'
    ]
    for i, (name, value) in enumerate(data):
        y = 6 + i * (bar_height + 6)
        bar_w = max(1.0, value / max_value * plot_w)
        parts.append(
            f'<text class="tick" x="{label_w - 8}" y="{y + bar_height * 0.7:.0f}" '
            f'text-anchor="end">{html.escape(name)}</text>'
        )
        parts.append(
            f'<rect x="{label_w}" y="{y}" width="{bar_w:.1f}" height="{bar_height}" '
            f'rx="3" fill="{PALETTE[i % len(PALETTE)]}"/>'
        )
        parts.append(
            f'<text class="tick" x="{label_w + bar_w + 8:.0f}" '
            f'y="{y + bar_height * 0.7:.0f}">{_fmt(value)}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def sparkline(values: Sequence[float], *, width: int = 120, height: int = 24) -> str:
    """Tiny inline trend line."""
    data = [float(v) for v in values]
    if len(data) < 2:
        return ""
    low, high = min(data), max(data)
    span = (high - low) or 1.0
    step = width / (len(data) - 1)
    points = " ".join(
        f"{i * step:.1f},{height - (v - low) / span * (height - 2) - 1:.1f}"
        for i, v in enumerate(data)
    )
    return (
        f'<svg class="spark" viewBox="0 0 {width} {height}" preserveAspectRatio="none">'
        f'<polyline fill="none" stroke="{PALETTE[0]}" stroke-width="1.5" points="{points}"/></svg>'
    )


_SPAN_CLASS = {
    "assistant_tool_call": "tok-call",
    "assistant_final": "tok-final",
    "assistant_text": "tok-text",
}


def token_strip(records: Sequence[dict[str, object]], *, max_tokens: int = 400) -> str:
    """Per-token divergence view.

    Each cell is one **stored, model-generated** token: colour intensity is the
    token's divergence, and the tooltip carries the span type, weight, teacher
    entropy and the two argmax tokens.  Nothing hidden or unstored is rendered.
    """
    rows = list(records)[:max_tokens]
    if not rows:
        return '<p class="muted">no token analysis recorded for this run</p>'
    losses = [float(r.get("token_loss") or 0.0) for r in rows]
    peak = max(losses) or 1.0
    cells = []
    for record, loss in zip(rows, losses):
        span = str(record.get("span_type", ""))
        css = _SPAN_CLASS.get(span, "tok-text")
        intensity = min(1.0, loss / peak)
        piece = record.get("token_piece") or f"id:{record.get('token_id')}"
        label = html.escape(str(piece).replace("\n", "\\n").replace(" ", "·"))
        entropy = record.get("teacher_entropy")
        tooltip = (
            f"pos {record.get('target_position')} | {span} | loss {loss:.4f}"
            f" | weight {record.get('weight')}"
            + (f" | teacher H {float(entropy):.3f}" if entropy is not None else "")
            + f" | teacher top {record.get('teacher_top_token')}"
            + f" | student top {record.get('student_top_token')}"
        )
        mismatch = (
            record.get("teacher_top_token") is not None
            and record.get("teacher_top_token") != record.get("student_top_token")
        )
        classes = f"tok {css}" + (" tok-mismatch" if mismatch else "")
        cells.append(
            f'<span class="{classes}" style="--i:{intensity:.3f}" '
            f'title="{html.escape(tooltip)}">{label}</span>'
        )
    return f'<div class="token-strip">{"".join(cells)}</div>'
