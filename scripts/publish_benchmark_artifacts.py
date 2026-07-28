#!/usr/bin/env python
"""Prepare portable benchmark JSON/Markdown/SVG artifacts from completed runs."""

from __future__ import annotations

import argparse
import hashlib
import html
import math
from pathlib import Path
from typing import Any

from miniverl.config import RunConfig
from miniverl.evaluation.benchmark import (
    portable_payload,
    render_benchmark_markdown,
    structured_diff,
)
from miniverl.evaluation.schema import BenchmarkResult
from miniverl.utils.runs import canonical_json, write_json, write_text


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
    rows = []
    for name in names:
        arms = grouped[name]
        success = [arm.strict_task_success_rate or 0.0 for arm in arms]
        seconds = [arm.train_seconds or 0.0 for arm in arms]
        rows.append(
            {
                "name": name,
                "success": success,
                "success_mean": sum(success) / len(success),
                "seconds": seconds,
                "seconds_mean": sum(seconds) / len(seconds),
            }
        )

    width = 1120
    height = 174 + len(rows) * 82
    label_x = 28
    success_x = 300
    success_w = 300
    time_x = 730
    time_w = 300
    max_seconds = max(row["seconds_mean"] for row in rows)
    time_ceiling = max(100, int(math.ceil(max_seconds / 100.0) * 100))
    colors = {
        "cold-start-only": "#6b7280",
        "sft-continued": "#3b82f6",
        "opd-raw-teacher": "#dc2626",
        "opd-privileged-context": "#f97316",
        "opd-protocol-sft-teacher": "#16a34a",
    }

    def line(text: str) -> str:
        return text + "\n"

    svg = line(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" '
        'aria-labelledby="benchmark-title benchmark-desc">'
    )
    svg += line('<title id="benchmark-title">miniVERL protocol-teacher benchmark</title>')
    svg += line(
        '<desc id="benchmark-desc">Strict held-out success and training time for five '
        "equal-optimizer-update arms over two prespecified seeds.</desc>"
    )
    svg += line(
        "<style>"
        "text{font-family:ui-sans-serif,system-ui,sans-serif;fill:#111827}"
        ".title{font-size:25px;font-weight:600}.sub{font-size:14px;fill:#4b5563}"
        ".head{font-size:15px;font-weight:600}.label{font-size:15px}"
        ".value{font-size:14px;font-weight:600}.axis{font-size:12px;fill:#6b7280}"
        ".grid{stroke:#d1d5db;stroke-width:1}.dot{fill:#fff;stroke-width:2}"
        "</style>"
    )
    svg += line('<rect width="100%" height="100%" fill="#ffffff"/>')
    svg += line('<text class="title" x="28" y="40">Protocol alignment prevents collapse</text>')
    svg += line(
        f'<text class="sub" x="28" y="66">schema v{result.schema_version} | '
        f"{len(result.seeds)} prespecified seeds | budget axis: "
        f"{html.escape(result.budget_axis or 'unreported')}</text>"
    )
    svg += line(f'<text class="head" x="{success_x}" y="110">Strict held-out success</text>')
    svg += line(f'<text class="head" x="{time_x}" y="110">Training time (seconds)</text>')
    for fraction in (0.0, 0.5, 1.0):
        x = success_x + success_w * fraction
        svg += line(f'<line class="grid" x1="{x:.1f}" y1="124" x2="{x:.1f}" y2="{height - 48}"/>')
        svg += line(
            f'<text class="axis" x="{x:.1f}" y="144" text-anchor="middle">'
            f"{fraction * 100:.0f}%</text>"
        )
    for fraction in (0.0, 0.5, 1.0):
        x = time_x + time_w * fraction
        svg += line(f'<line class="grid" x1="{x:.1f}" y1="124" x2="{x:.1f}" y2="{height - 48}"/>')
        svg += line(
            f'<text class="axis" x="{x:.1f}" y="144" text-anchor="middle">'
            f"{time_ceiling * fraction:.0f}</text>"
        )

    for index, row in enumerate(rows):
        y = 180 + index * 82
        color = colors.get(row["name"], "#4b5563")
        escaped_name = html.escape(row["name"])
        svg += line(
            f'<g data-arm="{escaped_name}" '
            f'data-success-mean="{row["success_mean"]:.6f}" '
            f'data-train-seconds-mean="{row["seconds_mean"]:.6f}">'
        )
        svg += line(f'<text class="label" x="{label_x}" y="{y + 5}">{escaped_name}</text>')

        success_end = success_x + success_w * row["success_mean"]
        svg += line(
            f'<rect x="{success_x}" y="{y - 13}" width="{success_end - success_x:.1f}" '
            f'height="26" rx="3" fill="{color}" opacity="0.78"/>'
        )
        for seed_index, value in enumerate(row["success"]):
            dot_x = success_x + success_w * value
            dot_y = y - 6 + seed_index * 12
            svg += line(
                f'<circle class="dot" cx="{dot_x:.1f}" cy="{dot_y}" r="4" stroke="{color}"/>'
            )
        svg += line(
            f'<text class="value" x="{success_x + success_w + 12}" y="{y + 5}">'
            f"{row['success_mean'] * 100:.0f}%</text>"
        )

        time_end = time_x + time_w * row["seconds_mean"] / time_ceiling
        svg += line(
            f'<rect x="{time_x}" y="{y - 13}" width="{time_end - time_x:.1f}" '
            f'height="26" rx="3" fill="{color}" opacity="0.78"/>'
        )
        for seed_index, value in enumerate(row["seconds"]):
            dot_x = time_x + time_w * value / time_ceiling
            dot_y = y - 6 + seed_index * 12
            svg += line(
                f'<circle class="dot" cx="{dot_x:.1f}" cy="{dot_y}" r="4" stroke="{color}"/>'
            )
        svg += line(
            f'<text class="value" x="{time_x + time_w + 12}" y="{y + 5}">'
            f"{row['seconds_mean']:.0f}s</text>"
        )
        svg += line("</g>")

    svg += line(
        f'<text class="axis" x="28" y="{height - 18}">Bars are seed means; hollow '
        f"dots are individual seeds {html.escape(str(result.seeds))}. "
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
