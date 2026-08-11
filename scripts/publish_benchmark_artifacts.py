#!/usr/bin/env python
"""Prepare portable benchmark JSON/Markdown/SVG artifacts from completed runs."""

from __future__ import annotations

import argparse
import hashlib
import html
import math
from pathlib import Path
from typing import Any, TypedDict

from miniverl.config import RunConfig
from miniverl.evaluation.benchmark import (
    portable_payload,
    render_benchmark_markdown,
    structured_diff,
)
from miniverl.evaluation.schema import BenchmarkResult
from miniverl.utils.runs import canonical_json, write_json, write_text


class _FigureRow(TypedDict):
    name: str
    success: list[float]
    success_mean: float
    seconds: list[float]
    seconds_mean: float
    optimizer_steps: list[int]


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _portable_config(path: Path) -> dict[str, Any]:
    return portable_payload(RunConfig.from_yaml(path).model_dump(mode="json"))


def prepare_result(source: Path, *, driver_version: str | None = None) -> BenchmarkResult:
    """Rebuild portable config provenance from the completed per-arm run configs."""
    result = BenchmarkResult.model_validate_json(source.read_text(encoding="utf-8"))
    if result.schema_version != 2 or result.common_resolved_config is None:
        raise ValueError("publishing requires a schema-v2 benchmark result")
    if driver_version is not None:
        gpu = result.hardware.get("gpu")
        if not isinstance(gpu, dict) or not gpu.get("available"):
            raise ValueError("cannot attach a driver version to a result without a measured GPU")
        gpu["driver_version"] = driver_version

    common = portable_payload(result.common_resolved_config)
    common_digest = _digest(common)
    result.common_resolved_config = common
    result.common_resolved_config_digest = common_digest
    result.controlled["digest"] = common_digest

    run_root = source.parent
    for arm in result.arms:
        config_path = run_root / arm.run_dir / "config.resolved.yaml"
        if not config_path.is_file():
            raise FileNotFoundError(f"missing resolved arm config: {config_path}")
        arm_config = _portable_config(config_path)
        arm.resolved_config_digest = _digest(arm_config)
        arm.structured_diff = structured_diff(common, arm_config)

    cold = result.cold_start
    if not isinstance(cold, dict):
        raise ValueError("schema-v2 benchmark has no cold-start provenance")
    cold_config = portable_payload(cold["resolved_config"])
    cold["resolved_config"] = cold_config
    cold["resolved_config_digest"] = _digest(cold_config)
    for checkpoint in cold.get("checkpoints", []):
        config_path = run_root / str(checkpoint["run_id"]) / "config.resolved.yaml"
        if not config_path.is_file():
            raise FileNotFoundError(f"missing resolved cold-start config: {config_path}")
        checkpoint["resolved_config_digest"] = _digest(_portable_config(config_path))

    result = BenchmarkResult.model_validate(portable_payload(result.model_dump(mode="json")))
    blob = canonical_json(result.model_dump(mode="json")).lower().replace("\\", "/")
    forbidden = ("/users/", "/home/", "onedrive")
    leaked = [needle for needle in forbidden if needle in blob]
    if leaked:
        raise ValueError(f"portable result still contains machine-local path markers: {leaked}")
    return result


def render_svg(result: BenchmarkResult, source_sha256: str) -> str:
    """Render success and training-time small multiples directly from result fields."""
    grouped = result.by_arm()
    names = list(grouped)
    rows: list[_FigureRow] = []
    for name in names:
        arms = grouped[name]
        success = [arm.strict_task_success_rate or 0.0 for arm in arms]
        seconds = [arm.train_seconds or 0.0 for arm in arms]
        optimizer_steps = [arm.optimizer_steps or 0 for arm in arms]
        rows.append(
            {
                "name": name,
                "success": success,
                "success_mean": sum(success) / len(success),
                "seconds": seconds,
                "seconds_mean": sum(seconds) / len(seconds),
                "optimizer_steps": optimizer_steps,
            }
        )
    display_order = {
        "cold-start-only": 0,
        "sft-continued": 1,
        "opd-protocol-sft-teacher": 2,
        "opd-raw-teacher": 3,
        "opd-privileged-context": 4,
    }
    rows.sort(key=lambda row: display_order.get(row["name"], len(display_order)))
    diagnostic_names = {"opd-raw-teacher", "opd-privileged-context"}
    diagnostic_count = sum(row["name"] in diagnostic_names for row in rows)
    diagnostic_start = next(
        (index for index, row in enumerate(rows) if row["name"] in diagnostic_names),
        None,
    )
    cold_start_seconds = next(
        (row["seconds_mean"] for row in rows if row["name"] == "cold-start-only"),
        None,
    )
    row_by_name = {row["name"]: row for row in rows}
    sft_row = row_by_name.get("sft-continued")
    protocol_opd_row = row_by_name.get("opd-protocol-sft-teacher")
    if (
        sft_row is not None
        and protocol_opd_row is not None
        and math.isclose(sft_row["success_mean"], protocol_opd_row["success_mean"])
        and sft_row["seconds_mean"] > 0
    ):
        continuation_ratio = protocol_opd_row["seconds_mean"] / sft_row["seconds_mean"]
        title = (
            "Saturated calculator task · same success, "
            f"{continuation_ratio:.1f}× OPD continuation time"
        )
    else:
        title = "Calculator task · strict success and continuation train time"
    trained_step_counts = {
        int(step)
        for row in rows
        if row["name"] != "cold-start-only"
        for step in row["optimizer_steps"]
    }
    if len(trained_step_counts) == 1:
        (trained_steps,) = trained_step_counts
        budget_description = f"equal budget: {trained_steps} continuation optimizer updates"
    else:
        budget_description = "equal continuation budget by optimizer updates"

    width = 1120
    diagnostic_gap = 36 if diagnostic_start is not None else 0
    height = 238 + len(rows) * 78 + diagnostic_gap
    label_x = 36
    success_x = 310
    success_w = 292
    time_x = 728
    time_w = 276
    max_seconds = max(row["seconds_mean"] for row in rows)
    time_ceiling = max(100, int(math.ceil(max_seconds / 100.0) * 100))
    colors = {
        "cold-start-only": "#94a3b8",
        "sft-continued": "#60a5fa",
        "opd-raw-teacher": "#fb7185",
        "opd-privileged-context": "#fbbf24",
        "opd-protocol-sft-teacher": "#34d399",
    }
    labels = {
        "cold-start-only": "Cold start",
        "sft-continued": "Continued SFT",
        "opd-raw-teacher": "Raw teacher (control)",
        "opd-privileged-context": "Privileged context (control)",
        "opd-protocol-sft-teacher": "OPD · protocol-aligned teacher",
    }

    def line(text: str) -> str:
        return text + "\n"

    svg = line(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" '
        'aria-labelledby="benchmark-title benchmark-desc">'
    )
    svg += line('<title id="benchmark-title">miniVERL protocol-teacher benchmark</title>')
    desc = (
        "Strict success on the v0.2 calculator test set and continuation train time "
        f"for equal-update arms over {len(result.seeds)} prespecified seeds."
    )
    if diagnostic_count:
        desc += (
            f" {diagnostic_count} completed negative-control arms without a protocol "
            "qualification gate are shown separately."
        )
    if cold_start_seconds is not None:
        desc += " The cold-start baseline has zero continuation updates."
    svg += line(f'<desc id="benchmark-desc">{desc}</desc>')
    svg += line(
        "<style>"
        "text{font-family:Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif;"
        "fill:#e6edf7}.title{font-size:25px;font-weight:760;letter-spacing:-.35px}"
        ".sub{font-size:13px;fill:#91a0b8}.head{font-size:14px;font-weight:700;fill:#dce7f7}"
        ".label{font-size:14px;font-weight:580}.value{font-size:13px;font-weight:730}"
        ".axis{font-size:11.5px;fill:#9aabc3}.note{font-size:13px;fill:#8fa0b9}"
        ".section{font-size:10.5px;font-weight:760;letter-spacing:1.15px;fill:#fda4af}"
        ".badge{font-size:9px;font-weight:760;letter-spacing:.55px;fill:#fecdd3}"
        ".panel{fill:url(#panel);stroke:#263752}.track{fill:#19263c}"
        ".grid{stroke:#2b3c58;stroke-width:1}.row{fill:#ffffff;fill-opacity:.018}"
        ".dot{fill:#08101f;stroke-width:2}.pill{fill:#0b1425;stroke-width:1.4}"
        "</style>"
    )
    svg += line(
        "<defs>"
        '<linearGradient id="background" x1="0" y1="0" x2="1" y2="1">'
        '<stop offset="0%" stop-color="#050816"/>'
        '<stop offset="58%" stop-color="#0a1224"/>'
        '<stop offset="100%" stop-color="#101a34"/>'
        "</linearGradient>"
        '<linearGradient id="panel" x1="0" y1="0" x2="0" y2="1">'
        '<stop offset="0%" stop-color="#111d34"/>'
        '<stop offset="100%" stop-color="#0b1426"/>'
        "</linearGradient>"
        '<radialGradient id="glow" cx="80%" cy="0%" r="70%">'
        '<stop offset="0%" stop-color="#4f46e5" stop-opacity=".24"/>'
        '<stop offset="100%" stop-color="#4f46e5" stop-opacity="0"/>'
        "</radialGradient>"
        "</defs>"
    )
    svg += line('<rect width="100%" height="100%" rx="18" fill="url(#background)"/>')
    svg += line('<rect width="100%" height="100%" rx="18" fill="url(#glow)"/>')
    svg += line(
        f'<rect x=".75" y=".75" width="{width - 1.5}" height="{height - 1.5}" rx="17.25" '
        'fill="none" stroke="#334663" stroke-width="1.5"/>'
    )
    svg += line(f'<text class="title" x="36" y="42">{html.escape(title)}</text>')
    svg += line(
        f'<text class="sub" x="36" y="68">Schema v{result.schema_version}  ·  '
        f"{len(result.seeds)} prespecified seeds  ·  "
        f"{html.escape(budget_description)}</text>"
    )
    panel_y = 96
    panel_h = height - 158
    svg += line(
        f'<rect class="panel" x="276" y="{panel_y}" width="394" height="{panel_h}" rx="14"/>'
    )
    svg += line(
        f'<rect class="panel" x="694" y="{panel_y}" width="390" height="{panel_h}" rx="14"/>'
    )
    svg += line(
        f'<text class="head" x="{success_x}" y="123">Strict success on v0.2 test set</text>'
    )
    svg += line(f'<text class="head" x="{time_x}" y="123">Continuation train time</text>')
    for fraction in (0.0, 0.5, 1.0):
        x = success_x + success_w * fraction
        svg += line(
            f'<text class="axis" x="{x:.1f}" y="151" text-anchor="middle">'
            f"{fraction * 100:.0f}%</text>"
        )
        svg += line(f'<line class="grid" x1="{x:.1f}" y1="166" x2="{x:.1f}" y2="{height - 74}"/>')
    for fraction in (0.0, 0.5, 1.0):
        x = time_x + time_w * fraction
        svg += line(
            f'<text class="axis" x="{x:.1f}" y="151" text-anchor="middle">'
            f"{time_ceiling * fraction:.0f}s</text>"
        )
        svg += line(f'<line class="grid" x1="{x:.1f}" y1="166" x2="{x:.1f}" y2="{height - 74}"/>')

    for index, row in enumerate(rows):
        is_diagnostic = row["name"] in diagnostic_names
        y = 204 + index * 78 + (diagnostic_gap if is_diagnostic else 0)
        color = colors.get(row["name"], "#4b5563")
        escaped_name = html.escape(row["name"])
        display_name = html.escape(labels.get(row["name"], row["name"]))
        if diagnostic_start is not None and index == diagnostic_start:
            svg += line(
                f'<rect x="24" y="{y - 57}" width="1072" height="28" rx="14" '
                'fill="#fb7185" fill-opacity=".075" stroke="#fb7185" stroke-opacity=".16"/>'
            )
            svg += line(
                f'<text class="section" x="38" y="{y - 39}">'
                "DIAGNOSTIC NEGATIVE CONTROLS · TEACHERS NOT PROTOCOL-QUALIFIED</text>"
            )
        svg += line(f'<rect class="row" x="24" y="{y - 30}" width="1072" height="60" rx="10"/>')
        svg += line(
            f'<g data-arm="{escaped_name}" '
            f'data-success-mean="{row["success_mean"]:.6f}" '
            f'data-train-seconds-mean="{row["seconds_mean"]:.6f}">'
        )
        label_y = y - 5 if is_diagnostic else y + 5
        svg += line(f'<text class="label" x="{label_x}" y="{label_y}">{display_name}</text>')
        if is_diagnostic:
            svg += line(
                f'<rect class="pill" x="{label_x}" y="{y + 3}" width="176" '
                f'height="18" rx="9" stroke="{color}" stroke-opacity=".65"/>'
            )
            svg += line(
                f'<text class="badge" x="{label_x + 88}" y="{y + 15.5}" '
                'text-anchor="middle">NEGATIVE CONTROL · UNGATED</text>'
            )

        success_end = success_x + success_w * row["success_mean"]
        svg += line(
            f'<rect class="track" x="{success_x}" y="{y - 10}" '
            f'width="{success_w}" height="20" rx="5"/>'
        )
        if row["success_mean"] > 0.0:
            svg += line(
                f'<rect x="{success_x}" y="{y - 10}" width="{success_end - success_x:.1f}" '
                f'height="20" rx="5" fill="{color}" opacity=".84"/>'
            )
        for seed_index, value in enumerate(row["success"]):
            dot_x = success_x + success_w * value
            dot_y = y - 5 + seed_index * 10
            svg += line(
                f'<circle class="dot" cx="{dot_x:.1f}" cy="{dot_y}" r="3.5" stroke="{color}"/>'
            )
        svg += line(
            f'<text class="value" x="{success_x + success_w + 12}" y="{y + 5}">'
            f"{row['success_mean'] * 100:.0f}%</text>"
        )

        if row["name"] == "cold-start-only":
            svg += line(
                f'<rect class="pill" x="{time_x + 12}" y="{y - 13}" width="190" '
                'height="26" rx="13" stroke="#64748b"/>'
            )
            svg += line(
                f'<text class="value" x="{time_x + 107}" y="{y + 4}" '
                'text-anchor="middle" fill="#94a3b8">0 CONTINUATION UPDATES</text>'
            )
        else:
            time_end = time_x + time_w * row["seconds_mean"] / time_ceiling
            svg += line(
                f'<rect class="track" x="{time_x}" y="{y - 10}" '
                f'width="{time_w}" height="20" rx="5"/>'
            )
            svg += line(
                f'<rect x="{time_x}" y="{y - 10}" width="{time_end - time_x:.1f}" '
                f'height="20" rx="5" fill="{color}" opacity=".84"/>'
            )
            for seed_index, value in enumerate(row["seconds"]):
                dot_x = time_x + time_w * value / time_ceiling
                dot_y = y - 5 + seed_index * 10
                svg += line(
                    f'<circle class="dot" cx="{dot_x:.1f}" cy="{dot_y}" r="3.5" stroke="{color}"/>'
                )
            svg += line(
                f'<text class="value" x="{time_x + time_w + 12}" y="{y + 5}">'
                f"{row['seconds_mean']:.0f}s</text>"
            )
        svg += line("</g>")

    svg += line(
        f'<text class="note" x="36" y="{height - 37}">'
        "Teacher preparation is not included in continuation bars.</text>"
    )
    svg += line(
        f'<text class="note" x="36" y="{height - 17}">Bars = seed means · dots = seeds '
        f"{html.escape(str(result.seeds))} · "
        f"Source JSON SHA-256 {source_sha256[:16]}</text>"
    )
    svg += "</svg>\n"
    return svg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--svg", type=Path, required=True)
    parser.add_argument(
        "--driver-version",
        help="driver version measured separately when torch did not report it",
    )
    args = parser.parse_args()

    result = prepare_result(args.source, driver_version=args.driver_version)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.results_dir / f"{result.name}.json"
    markdown_path = args.results_dir / f"{result.name}.md"
    write_json(json_path, result.model_dump(mode="json"))
    write_text(markdown_path, render_benchmark_markdown(result))
    source_sha = hashlib.sha256(json_path.read_bytes()).hexdigest()
    write_text(args.svg, render_svg(result, source_sha))
    print(f"wrote {json_path}, {markdown_path}, and {args.svg}")
    print(f"published JSON SHA-256 {source_sha}")


if __name__ == "__main__":
    main()
