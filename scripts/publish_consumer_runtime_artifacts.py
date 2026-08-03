#!/usr/bin/env python3
"""Validate and render the frozen consumer-runtime v1 benchmark artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "benchmarks/results/consumer-runtime-v1.json"
OUTPUT = ROOT / "docs/consumer-runtime-v1-pareto.svg"
FROZEN_CALCULATOR = ROOT / "benchmarks/results/gpu-calc-hard-equal-update-v2.json"
FROZEN_CALCULATOR_SHA256 = "53fc1d4d5b7adee09618d77ad62d4086ba56b78569832d6fc7c3bcd5c2695bbc"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_result(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("name") != "consumer-runtime-v1":
        raise ValueError("not the consumer-runtime-v1 result")
    if payload.get("measurement_status") != "measured_final":
        raise ValueError("consumer-runtime-v1 result is not a valid final measurement")
    invariants = payload.get("workload_invariants", {})
    if not (
        invariants.get("trajectory_digests_identical")
        and invariants.get("teacher_target_digests_identical")
    ):
        raise ValueError("consumer-runtime-v1 workload invariants failed")
    if not payload.get("equivalence_gate", {}).get("passed"):
        raise ValueError("consumer-runtime-v1 equivalence gate failed")
    completed = [cell for cell in payload.get("cells", []) if cell.get("status") == "completed"]
    keys = {(cell["runtime"], str(cell["batch_size"])) for cell in completed}
    expected = {
        (runtime, batch)
        for runtime in ("dual_model", "shared_backbone")
        for batch in ("1", "2", "4", "auto")
    }
    if keys != expected:
        raise ValueError(f"consumer-runtime-v1 matrix changed: {sorted(keys)}")
    if _sha256(FROZEN_CALCULATOR) != FROZEN_CALCULATOR_SHA256:
        raise ValueError("immutable calculator benchmark hash changed")
    return payload


def _cell_map(payload: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (cell["runtime"], str(cell["batch_size"])): cell
        for cell in payload["cells"]
        if cell["status"] == "completed"
    }


def render_pareto(payload: dict[str, Any], source_sha256: str) -> str:
    cells = _cell_map(payload)
    width, height = 1120, 620
    left, right, top, bottom = 88.0, 790.0, 162.0, 500.0
    x_min, x_max = 2.0, 4.8
    y_min, y_max = 1.8, 4.1

    def x(value: float) -> float:
        return left + (value - x_min) * (right - left) / (x_max - x_min)

    def y(value: float) -> float:
        return bottom - (value - y_min) * (bottom - top) / (y_max - y_min)

    def gib(cell: dict[str, Any]) -> float:
        return float(cell["peak_reserved_bytes"]) / 2**30

    def throughput(cell: dict[str, Any]) -> float:
        return float(cell["trajectories_per_second"])

    style = (
        "text{font-family:Inter,'Segoe UI',sans-serif;fill:#e8eefc}"
        ".title{font-size:30px;font-weight:760}.sub{font-size:16px;fill:#9fb0cc}"
        ".axis{font-size:14px;fill:#91a1bd}.point{font-size:14px;font-weight:750}"
        ".metric{font-size:25px;font-weight:760}.small{font-size:14px;fill:#9fb0cc}"
        ".legend{font-size:15px;font-weight:650}.foot{font-size:15px;fill:#91a1bd}"
    )
    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Batch-4 is the Pareto knee on one RTX 4080</title>',
        '<desc id="desc">Throughput versus peak reserved CUDA memory for dual-model and '
        "shared-backbone runtimes at physical trajectory batch sizes 1, 2, 4 and auto. "
        "Shared batch-4 reaches 3.48 trajectories per second at 2.23 GiB; dual batch-4 "
        "reaches 3.87 trajectories per second at 3.04 GiB.</desc>",
        '<rect width="1120" height="620" rx="24" fill="#070b17"/>',
        '<rect x="20" y="20" width="1080" height="580" rx="20" fill="#0b1224" stroke="#20304f"/>',
        f"<style>{style}</style>",
        '<text class="title" x="52" y="66">Batch-4 is the Pareto knee on one RTX 4080</text>',
        '<text class="sub" x="52" y="96">8-trajectory strict OPD update · Qwen3-0.6B · '
        "NF4 weights / FP32 compute · medians over 3 updates</text>",
        '<circle cx="550" cy="129" r="6" fill="#fb7185"/><text class="legend" x="564" y="134">dual model</text>',
        '<circle cx="664" cy="129" r="6" fill="#22d3ee"/><text class="legend" x="678" y="134">shared backbone</text>',
    ]

    for tick in (2.0, 2.5, 3.0, 3.5, 4.0, 4.5):
        px = x(tick)
        body.extend(
            [
                f'<line x1="{px:.1f}" y1="{top:.1f}" x2="{px:.1f}" y2="{bottom:.1f}" stroke="#22304b"/>',
                f'<text class="axis" x="{px:.1f}" y="525" text-anchor="middle">{tick:.1f}</text>',
            ]
        )
    for tick in (2.0, 2.5, 3.0, 3.5, 4.0):
        py = y(tick)
        body.extend(
            [
                f'<line x1="{left:.1f}" y1="{py:.1f}" x2="{right:.1f}" y2="{py:.1f}" stroke="#22304b"/>',
                f'<text class="axis" x="72" y="{py + 5:.1f}" text-anchor="end">{tick:.1f}</text>',
            ]
        )
    body.extend(
        [
            '<text class="axis" x="439" y="554" text-anchor="middle">peak reserved CUDA memory (GiB) →</text>',
            '<text class="axis" x="24" y="335" transform="rotate(-90 24 335)" text-anchor="middle">trajectories / second →</text>',
        ]
    )

    series = (
        (
            "dual_model",
            "#fb7185",
            {"1": (-33, -13), "2": (-8, -14), "4": (11, -12), "auto": (10, 20)},
        ),
        (
            "shared_backbone",
            "#22d3ee",
            {"1": (-36, 21), "2": (-36, -13), "4": (-7, -14), "auto": (10, 20)},
        ),
    )
    for runtime, color, offsets in series:
        points = []
        for batch in ("1", "2", "4", "auto"):
            cell = cells[(runtime, batch)]
            points.append(f"{x(gib(cell)):.1f},{y(throughput(cell)):.1f}")
        body.append(
            f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" '
            'stroke-width="3" stroke-linecap="round" stroke-linejoin="round" opacity="0.7"/>'
        )
        for batch in ("1", "2", "4", "auto"):
            cell = cells[(runtime, batch)]
            px, py = x(gib(cell)), y(throughput(cell))
            dx, dy = offsets[batch]
            label = "auto" if batch == "auto" else f"b{batch}"
            body.extend(
                [
                    f'<circle cx="{px:.1f}" cy="{py:.1f}" r="8" fill="#0b1224" stroke="{color}" stroke-width="4"/>',
                    f'<text class="point" x="{px + dx:.1f}" y="{py + dy:.1f}" fill="{color}">{escape(label)}</text>',
                ]
            )

    dual_one = cells[("dual_model", "1")]
    dual_four = cells[("dual_model", "4")]
    shared_one = cells[("shared_backbone", "1")]
    shared_four = cells[("shared_backbone", "4")]
    dual_speedup = throughput(dual_four) / throughput(dual_one)
    shared_speedup = throughput(shared_four) / throughput(shared_one)
    body.extend(
        [
            '<rect x="824" y="162" width="244" height="148" rx="16" fill="#101b33" stroke="#26446b"/>',
            '<text class="legend" x="846" y="190" fill="#22d3ee">SHARED · BATCH 4</text>',
            f'<text class="metric" x="846" y="228">{throughput(shared_four):.2f} traj/s</text>',
            f'<text class="small" x="846" y="257">{gib(shared_four):.2f} GiB reserved</text>',
            f'<text class="small" x="846" y="282">{shared_speedup:.2f}× vs sequential</text>',
            '<rect x="824" y="326" width="244" height="148" rx="16" fill="#101b33" stroke="#4d3048"/>',
            '<text class="legend" x="846" y="354" fill="#fb7185">DUAL · BATCH 4</text>',
            f'<text class="metric" x="846" y="392">{throughput(dual_four):.2f} traj/s</text>',
            f'<text class="small" x="846" y="421">{gib(dual_four):.2f} GiB reserved</text>',
            f'<text class="small" x="846" y="446">{dual_speedup:.2f}× vs sequential</text>',
            '<text class="small" x="824" y="502">Same batch, dual ↔ shared</text>',
            '<text class="legend" x="824" y="526">loss + gradients: exact</text>',
            '<text class="legend" x="824" y="548">update logits: exact</text>',
            '<line x1="52" y1="568" x2="1068" y2="568" stroke="#20304f"/>',
            '<text class="foot" x="52" y="591">Higher and left is better · auto = all 8 padded trajectories · '
            f"source SHA-256 {escape(source_sha256[:16])}</text>",
        ]
    )
    body.append("</svg>\n")
    return "".join(body)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, default=RESULT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    payload = _load_result(args.result)
    rendered = render_pareto(payload, _sha256(args.result))
    if args.check:
        if not args.output.exists() or args.output.read_text(encoding="utf-8") != rendered:
            raise SystemExit(f"generated artifact is stale: {args.output}")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
