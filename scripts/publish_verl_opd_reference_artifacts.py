"""Render the measured single-GPU OPD developer workload systems figure."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "benchmarks/results/rtx4080-verl-opd-developer-v1.json"
FIGURE = ROOT / "docs/verl-opd-reference-workload.svg"
MOBILE_FIGURE = ROOT / "docs/verl-opd-reference-workload-mobile.svg"


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("kind") != "single_gpu_opd_developer_workload":
        raise ValueError("not a single-GPU OPD developer workload record")
    if payload.get("status") != "measured":
        raise ValueError("developer workload is not measured")
    if payload["scientific_scope"] != {
        "runtime_correctness_only": True,
        "alignment_quality_evaluated": False,
        "task_quality_evaluated": False,
        "opd_beats_sft_dpo_or_kd_claimed": False,
    }:
        raise ValueError("developer workload scientific scope drifted")
    return payload


def _text(x: int, y: int, value: str, class_name: str = "label", anchor: str = "start") -> str:
    return (
        f'<text x="{x}" y="{y}" class="{class_name}" text-anchor="{anchor}">'
        f"{html.escape(value)}</text>"
    )


def render(payload: dict[str, Any]) -> str:
    measurements = payload["measurements"]
    phases = measurements["steady_state_median_seconds"]
    rates = measurements["steady_state_median_throughput"]
    peak = float(measurements["peak_reserved_gib"])
    limit = float(payload["resource_contract"]["peak_reserved_limit_gib"])
    phase_rows = [
        ("Actor rollout", float(phases["rollout"]), "#56B4E9"),
        ("Teacher scoring", float(phases["teacher_scoring"]), "#E69F00"),
        ("Actor update", float(phases["actor_update"]), "#009E73"),
    ]
    rate_rows = [
        ("Rollout tokens/s", float(rates["rollout_tokens_per_second"]), "#56B4E9"),
        ("Teacher positions/s", float(rates["teacher_scored_positions_per_second"]), "#E69F00"),
        ("Update positions/s", float(rates["update_positions_per_second"]), "#009E73"),
    ]
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1120 690" role="img" aria-labelledby="title desc">',
        '<title id="title">RTX 4080 verl-style OPD developer workload</title>',
        '<desc id="desc">Measured median phase time and throughput plus peak reserved VRAM for eight QLoRA optimizer updates over 32 distinct prompts. Resume matched uninterrupted adapter, optimizer, trajectories and training state. No task quality was evaluated.</desc>',
        """<style>
        text { font-family: Inter, Segoe UI, Arial, sans-serif; }
        .title { fill:#f8fafc; font-size:30px; font-weight:750; }
        .subtitle { fill:#a9b8d4; font-size:17px; }
        .panel-title { fill:#f8fafc; font-size:19px; font-weight:700; }
        .label { fill:#dbe7fa; font-size:16px; }
        .value { fill:#ffffff; font-size:16px; font-weight:700; }
        .note { fill:#a9b8d4; font-size:15px; }
        .small { fill:#a9b8d4; font-size:14px; }
        </style>""",
        '<rect width="1120" height="690" rx="22" fill="#08111f"/>',
        '<rect x="28" y="28" width="1064" height="634" rx="18" fill="#0f1b2d" stroke="#283b58"/>',
        _text(58, 76, "ONE RTX 4080 · QWEN3 0.6B → 1.7B · FORWARD-KL TOP-K 32", "small"),
        _text(58, 116, "A practical single-GPU OPD workload, measured end to end", "title"),
        _text(
            58,
            148,
            "32 distinct prompts consumed · 64 response tokens · logical batch 4 · 8 optimizer updates",
            "subtitle",
        ),
        '<rect x="50" y="180" width="500" height="314" rx="14" fill="#111f34" stroke="#2b4263"/>',
        _text(74, 216, "Median steady-state phase time", "panel-title"),
        _text(526, 216, "seconds", "small", "end"),
    ]
    max_phase = max(value for _, value, _ in phase_rows)
    for index, (label, value, color) in enumerate(phase_rows):
        y = 260 + index * 76
        width = 300 * value / max_phase
        parts.extend(
            [
                _text(74, y, label, "label"),
                f'<rect x="210" y="{y - 20}" width="300" height="24" rx="6" fill="#1b2b44"/>',
                f'<rect x="210" y="{y - 20}" width="{width:.2f}" height="24" rx="6" fill="{color}"/>',
                f'<circle cx="{210 + width:.2f}" cy="{y - 8}" r="6" fill="#f8fafc" stroke="{color}" stroke-width="3"/>',
                _text(526, y, f"{value:.4f}", "value", "end"),
            ]
        )
    parts.extend(
        [
            _text(74, 472, "Direct labels; bars share a zero baseline.", "note"),
            '<rect x="570" y="180" width="500" height="314" rx="14" fill="#111f34" stroke="#2b4263"/>',
            _text(594, 216, "Median steady-state throughput", "panel-title"),
            _text(1046, 216, "items / second", "small", "end"),
        ]
    )
    max_rate = max(value for _, value, _ in rate_rows)
    for index, (label, value, color) in enumerate(rate_rows):
        y = 260 + index * 76
        width = 278 * value / max_rate
        parts.extend(
            [
                _text(594, y, label, "label"),
                f'<rect x="750" y="{y - 20}" width="278" height="24" rx="6" fill="#1b2b44"/>',
                f'<rect x="750" y="{y - 20}" width="{width:.2f}" height="24" rx="6" fill="{color}"/>',
                f'<path d="M {750 + width:.2f} {y - 19} l 7 11 l -7 11 l -7 -11 z" fill="#f8fafc" stroke="{color}" stroke-width="2"/>',
                _text(1046, y, f"{value:.2f}", "value", "end"),
            ]
        )
    gauge_width = 760 * peak / limit
    parts.extend(
        [
            _text(594, 472, "Token and selected-position rates are labelled separately.", "note"),
            '<rect x="50" y="516" width="1020" height="118" rx="14" fill="#111f34" stroke="#2b4263"/>',
            _text(74, 550, "Peak reserved VRAM", "panel-title"),
            _text(1046, 550, f"{peak:.4f} GiB / {limit:.1f} GiB release gate", "value", "end"),
            '<rect x="74" y="570" width="760" height="24" rx="7" fill="#1b2b44"/>',
            f'<rect x="74" y="570" width="{gauge_width:.2f}" height="24" rx="7" fill="#CC79A7"/>',
            f'<path d="M {74 + gauge_width:.2f} 566 l 8 16 l -8 16 l -8 -16 z" fill="#f8fafc" stroke="#CC79A7" stroke-width="2"/>',
            _text(852, 590, "0 OOM downshifts", "value"),
            _text(
                74,
                620,
                "Resume: adapter + optimizer + trajectories + training state matched exactly · systems evidence only · no quality endpoint",
                "note",
            ),
            "</svg>",
        ]
    )
    return "\n".join(parts) + "\n"


def render_mobile(payload: dict[str, Any]) -> str:
    measurements = payload["measurements"]
    phases = measurements["steady_state_median_seconds"]
    rates = measurements["steady_state_median_throughput"]
    peak = float(measurements["peak_reserved_gib"])
    limit = float(payload["resource_contract"]["peak_reserved_limit_gib"])
    phase_rows = [
        ("Actor rollout", float(phases["rollout"]), "#56B4E9"),
        ("Teacher scoring", float(phases["teacher_scoring"]), "#E69F00"),
        ("Actor update", float(phases["actor_update"]), "#009E73"),
    ]
    rate_rows = [
        ("Rollout tokens/s", float(rates["rollout_tokens_per_second"]), "#56B4E9"),
        ("Teacher positions/s", float(rates["teacher_scored_positions_per_second"]), "#E69F00"),
        ("Update positions/s", float(rates["update_positions_per_second"]), "#009E73"),
    ]
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 390 920" role="img" aria-labelledby="title desc">',
        '<title id="title">RTX 4080 verl-style OPD developer workload</title>',
        '<desc id="desc">Mobile layout of measured median phase time and throughput plus peak reserved VRAM for eight QLoRA updates over 32 distinct prompts. Resume matched uninterrupted training artifacts. No task quality was evaluated.</desc>',
        """<style>
        text { font-family: Inter, Segoe UI, Arial, sans-serif; }
        .title { fill:#f8fafc; font-size:20px; font-weight:750; }
        .subtitle { fill:#a9b8d4; font-size:13px; }
        .panel-title { fill:#f8fafc; font-size:16px; font-weight:700; }
        .label { fill:#dbe7fa; font-size:14px; }
        .value { fill:#ffffff; font-size:14px; font-weight:700; }
        .note { fill:#a9b8d4; font-size:13px; }
        .small { fill:#a9b8d4; font-size:12px; }
        </style>""",
        '<rect width="390" height="920" rx="18" fill="#08111f"/>',
        '<rect x="14" y="14" width="362" height="892" rx="14" fill="#0f1b2d" stroke="#283b58"/>',
        _text(30, 48, "ONE RTX 4080 · QWEN3 0.6B → 1.7B", "small"),
        _text(30, 78, "Single-GPU OPD workload", "title"),
        _text(30, 102, "32 prompts · 64 response tokens · 8 updates", "subtitle"),
        '<rect x="26" y="126" width="338" height="260" rx="12" fill="#111f34" stroke="#2b4263"/>',
        _text(42, 158, "Median phase time", "panel-title"),
        _text(348, 158, "seconds", "small", "end"),
    ]
    max_phase = max(value for _, value, _ in phase_rows)
    for index, (label, value, color) in enumerate(phase_rows):
        y = 204 + index * 58
        width = 174 * value / max_phase
        parts.extend(
            [
                _text(42, y, label, "label"),
                f'<rect x="154" y="{y - 17}" width="174" height="20" rx="5" fill="#1b2b44"/>',
                f'<rect x="154" y="{y - 17}" width="{width:.2f}" height="20" rx="5" fill="{color}"/>',
                _text(348, y, f"{value:.4f}", "value", "end"),
            ]
        )
    parts.extend(
        [
            _text(42, 364, "Shared zero baseline", "note"),
            '<rect x="26" y="404" width="338" height="260" rx="12" fill="#111f34" stroke="#2b4263"/>',
            _text(42, 436, "Median throughput", "panel-title"),
            _text(348, 436, "items / second", "small", "end"),
        ]
    )
    max_rate = max(value for _, value, _ in rate_rows)
    for index, (label, value, color) in enumerate(rate_rows):
        y = 482 + index * 58
        width = 174 * value / max_rate
        parts.extend(
            [
                _text(42, y, label, "label"),
                f'<rect x="154" y="{y - 17}" width="174" height="20" rx="5" fill="#1b2b44"/>',
                f'<rect x="154" y="{y - 17}" width="{width:.2f}" height="20" rx="5" fill="{color}"/>',
                _text(348, y, f"{value:.2f}", "value", "end"),
            ]
        )
    parts.extend(
        [
            _text(42, 642, "Tokens and positions labelled separately", "note"),
            '<rect x="26" y="682" width="338" height="194" rx="12" fill="#111f34" stroke="#2b4263"/>',
            _text(42, 716, "Peak reserved VRAM", "panel-title"),
            _text(348, 744, f"{peak:.4f} / {limit:.1f} GiB gate", "value", "end"),
            '<rect x="42" y="764" width="290" height="22" rx="6" fill="#1b2b44"/>',
            f'<rect x="42" y="764" width="{290 * peak / limit:.2f}" height="22" rx="6" fill="#CC79A7"/>',
            _text(42, 818, "0 OOM downshifts · exact resume", "value"),
            _text(42, 848, "Systems evidence only · no quality endpoint", "note"),
            "</svg>",
        ]
    )
    return "\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=RESULT)
    parser.add_argument("--out", type=Path, default=FIGURE)
    parser.add_argument("--mobile-out", type=Path, default=MOBILE_FIGURE)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    payload = _load(args.result)
    rendered = render(payload)
    mobile = render_mobile(payload)
    if args.check:
        if not args.out.is_file() or args.out.read_text(encoding="utf-8") != rendered:
            raise SystemExit(f"generated artifact is stale: {args.out}")
        if not args.mobile_out.is_file() or args.mobile_out.read_text(encoding="utf-8") != mobile:
            raise SystemExit(f"generated artifact is stale: {args.mobile_out}")
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(rendered, encoding="utf-8", newline="\n")
    args.mobile_out.write_text(mobile, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
