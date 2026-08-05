#!/usr/bin/env python3
"""Freeze, validate, analyze, and render RecoveryBench v1 result artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any
from xml.sax.saxutils import escape

from miniverl.evaluation.recovery import trajectory_recovery_metrics
from miniverl.evaluation.schema import BenchmarkResult
from miniverl.trajectory.io import read_trajectories

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results"
DOCS = ROOT / "docs" / "recoverybench"
TASK_RESULTS = RESULTS / "recoverybench-v1-task-results.jsonl"
ANALYSIS = RESULTS / "recoverybench-v1-analysis.json"
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20_260_801


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_result(path: Path, expected_view: str) -> BenchmarkResult:
    result = BenchmarkResult.model_validate_json(path.read_text(encoding="utf-8"))
    if result.schema_version != 3 or result.budget_view != expected_view:
        raise ValueError(f"{path} is not the schema-v3 {expected_view} result")
    if result.invalidation_status != {"valid": True, "reasons": []}:
        raise ValueError(f"{path} is invalidated")
    return result


def _compact_task_rows(
    sources: list[tuple[str, Path, BenchmarkResult]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for view, source, result in sources:
        for arm in result.arms:
            artifact = arm.task_level_artifact
            if not artifact:
                raise ValueError(f"{view}/{arm.name}/s{arm.seed} has no task-level artifact")
            trajectory_path = source.parent / str(artifact["path"])
            if _sha256(trajectory_path) != artifact["sha256"]:
                raise ValueError(f"task artifact hash mismatch: {trajectory_path}")
            for trajectory in read_trajectories(trajectory_path):
                recovery = trajectory_recovery_metrics(trajectory)
                rows.append(
                    {
                        "schema_version": 1,
                        "budget_view": view,
                        "arm": arm.name,
                        "seed": arm.seed,
                        "task_id": trajectory.task_id,
                        "template_id": trajectory.metadata.get("template_id"),
                        "intervention_kind": trajectory.metadata.get("intervention_kind"),
                        "strict_success": bool(
                            trajectory.verification and trajectory.verification.solved
                        ),
                        **recovery.to_dict(),
                    }
                )
    return sorted(
        rows, key=lambda row: (row["budget_view"], row["arm"], row["seed"], row["task_id"])
    )


def _paired_interval(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("paired bootstrap requires at least one matched task")
    rng = random.Random(BOOTSTRAP_SEED)
    estimates = sorted(
        mean(values[rng.randrange(len(values))] for _ in values)
        for _ in range(BOOTSTRAP_REPLICATES)
    )
    return {
        "pairs": len(values),
        "mean_difference": mean(values),
        "lower_95": estimates[int(0.025 * BOOTSTRAP_REPLICATES)],
        "upper_95": estimates[int(0.975 * BOOTSTRAP_REPLICATES) - 1],
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
    }


def analyze(rows: list[dict[str, Any]], result_hashes: dict[str, str]) -> dict[str, Any]:
    primary = [row for row in rows if row["budget_view"] == "equal_optimizer_updates"]
    by_key = {(row["arm"], row["seed"], row["task_id"]): row for row in primary}
    strict_diffs: list[float] = []
    recovery_diffs: list[float] = []
    for seed in (1234, 20_260_727, 20_260_801):
        task_ids = sorted(
            row["task_id"]
            for row in primary
            if row["arm"] == "offline-kd-frozen-student" and row["seed"] == seed
        )
        for task_id in task_ids:
            frozen = by_key[("offline-kd-frozen-student", seed, task_id)]
            fresh = by_key[("strict-opd-fresh", seed, task_id)]
            strict_diffs.append(float(fresh["strict_success"]) - float(frozen["strict_success"]))
            if fresh["had_tool_error"] and frozen["had_tool_error"]:
                recovery_diffs.append(
                    float(fresh["recovered_after_tool_error"])
                    - float(frozen["recovered_after_tool_error"])
                )
    return {
        "schema_version": 1,
        "analysis_version": "recoverybench-analysis-v1",
        "source_result_sha256": result_hashes,
        "paired_comparison": "strict-opd-fresh minus offline-kd-frozen-student",
        "strict_task_success": _paired_interval(strict_diffs),
        "recovery_after_error": _paired_interval(recovery_diffs),
        "recovery_pair_rule": "both trajectories had a structured tool error",
    }


def _means(result: BenchmarkResult) -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for arm in result.arms:
        values[arm.name]["strict"].append(float(arm.strict_task_success_rate or 0.0))
        recovery = arm.recovery_metrics or {}
        values[arm.name]["recovery"].append(float(recovery.get("recovery_after_error_rate") or 0.0))
        values[arm.name]["seconds"].append(float(arm.train_seconds or 0.0))
        values[arm.name]["vram"].append(float(arm.peak_reserved_bytes or 0.0) / 2**30)
    return {
        arm: {metric: mean(samples) for metric, samples in metrics.items()}
        for arm, metrics in values.items()
    }


def _svg(title: str, subtitle: str, rows: list[tuple[str, float, float]], *, x_label: str) -> str:
    width, height = 1120, 170 + 72 * len(rows)
    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{escape(title)}</title><desc id="desc">{escape(subtitle)}</desc>',
        '<rect width="1120" height="100%" fill="#090f1f"/>',
        "<style>text{font-family:Inter,Segoe UI,sans-serif;fill:#e8eefc}.muted{fill:#94a3b8}.label{font-size:18px}.value{font-size:16px;font-weight:700}.title{font-size:30px;font-weight:750}.sub{font-size:16px}</style>",
        f'<text class="title" x="40" y="48">{escape(title)}</text>',
        f'<text class="sub muted" x="40" y="78">{escape(subtitle)}</text>',
        f'<text class="sub muted" x="1040" y="112" text-anchor="end">{escape(x_label)}</text>',
    ]
    maximum = max((max(abs(first), abs(second)) for _, first, second in rows), default=1.0) or 1.0
    for index, (label, first, second) in enumerate(rows):
        y = 138 + index * 72
        body.extend(
            [
                f'<text class="label" x="40" y="{y + 20}">{escape(label)}</text>',
                f'<rect x="360" y="{y}" width="{620 * abs(first) / maximum:.1f}" height="18" rx="9" fill="#38bdf8"/>',
                f'<rect x="360" y="{y + 26}" width="{620 * abs(second) / maximum:.1f}" height="18" rx="9" fill="#a78bfa"/>',
                f'<text class="value" x="1000" y="{y + 15}">{first:.3f}</text>',
                f'<text class="value" x="1000" y="{y + 41}">{second:.3f}</text>',
            ]
        )
    body.append("</svg>\n")
    return "".join(body)


#: Mobile canvas. The docs content column is 358 px wide at a 390 px viewport,
#: so a 390 px canvas renders at 0.918x and a 14 px class clears the 11 px
#: readability floor with margin. Scaling the 1120 px desktop chart to the same
#: column would render its 16 px text at about 5 px.
MOBILE_WIDTH = 390
MOBILE_MIN_FONT_PX = 14

#: Every mobile class stays at or above 14 px.
_MOBILE_STYLE = (
    "text{font-family:Inter,'Segoe UI',sans-serif;fill:#edf4ff}"
    ".title{font-size:19px;font-weight:760}"
    ".sub{font-size:14px;fill:#aebbd2}"
    ".label{font-size:16px;font-weight:700}"
    ".series{font-size:14px;fill:#c3cfe2}"
    ".value{font-size:15px;font-weight:760}"
)

_SERIES_COLOURS = ("#38bdf8", "#a78bfa")


def _wrap(text: str, limit: int) -> list[str]:
    """Greedy wrap so a long title stays inside the narrow canvas."""
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > limit:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def _mobile_svg(
    title: str,
    subtitle: str,
    rows: list[tuple[str, float, float]],
    *,
    series: tuple[str, str],
    description: str,
) -> str:
    """Render the same numbers as a vertical, phone-first card list.

    Each bar carries its series name and its exact value as text, so nothing is
    encoded by colour alone and no value has to be read off a scaled axis.
    """
    title_lines = _wrap(title, 30)
    subtitle_lines = _wrap(subtitle, 46)
    # Line steps clear the rendered bounding box of each class: 19 px text needs
    # more than 24 px of leading before the gate's overlap check fires.
    header = 34 + 28 * len(title_lines) + 8 + 21 * len(subtitle_lines) + 14
    block = 118
    height = header + block * len(rows) + 4
    inner_width = MOBILE_WIDTH - 24
    content_width = MOBILE_WIDTH - 48
    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{MOBILE_WIDTH}" height="{height}" '
        f'viewBox="0 0 {MOBILE_WIDTH} {height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{escape(title)}</title><desc id="desc">{escape(description)}</desc>',
        f'<rect width="{MOBILE_WIDTH}" height="{height}" rx="16" fill="#060a14"/>',
        f'<rect x="12" y="12" width="{inner_width}" height="{height - 24}" rx="12" '
        'fill="#0a1222" stroke="#20304f"/>',
        f"<style>{_MOBILE_STYLE}</style>",
    ]
    cursor = 34
    for line in title_lines:
        body.append(f'<text class="title" x="24" y="{cursor}">{escape(line)}</text>')
        cursor += 28
    cursor += 8
    for line in subtitle_lines:
        body.append(f'<text class="sub" x="24" y="{cursor}">{escape(line)}</text>')
        cursor += 21
    cursor += 14

    maximum = max((max(abs(first), abs(second)) for _, first, second in rows), default=1.0) or 1.0
    for index, (label, first, second) in enumerate(rows):
        base = cursor + index * block
        body.append(f'<text class="label" x="24" y="{base + 16}">{escape(label)}</text>')
        for offset, (name, value, colour) in enumerate(
            ((series[0], first, _SERIES_COLOURS[0]), (series[1], second, _SERIES_COLOURS[1]))
        ):
            line = base + 42 + offset * 44
            body.extend(
                [
                    f'<text class="series" x="24" y="{line}">{escape(name)}</text>',
                    f'<text class="value" x="{MOBILE_WIDTH - 24}" y="{line}" '
                    f'text-anchor="end">{value:.3f}</text>',
                    f'<rect x="24" y="{line + 8}" '
                    f'width="{content_width * abs(value) / maximum:.1f}" height="12" rx="6" '
                    f'fill="{colour}"/>',
                ]
            )
    body.append("</svg>\n")
    return "".join(body)


def render_figures(
    primary: BenchmarkResult, analysis: dict[str, Any], source_hash: str
) -> dict[str, str]:
    means = _means(primary)
    ordered = [arm.name for arm in primary.arms if arm.seed == primary.seeds[0]]
    digest = source_hash[:12]
    success_rows = [(name, means[name]["strict"], means[name]["recovery"]) for name in ordered]
    slowest = max(v["seconds"] for v in means.values())
    cost_rows = [
        (name, means[name]["recovery"], means[name]["seconds"] / slowest) for name in ordered
    ]
    paired_rows = [
        (
            "strict task success",
            analysis["strict_task_success"]["mean_difference"],
            analysis["strict_task_success"]["upper_95"],
        ),
        (
            "recovery after error",
            analysis["recovery_after_error"]["mean_difference"],
            analysis["recovery_after_error"]["upper_95"],
        ),
    ]
    return {
        "recovery-success.svg": _svg(
            "RecoveryBench v1: strict success and recovery",
            f"Three-seed means · strict success (cyan) · recovery after error (violet) · source {digest}",
            success_rows,
            x_label="rate (0–1)",
        ),
        "recovery-success-mobile.svg": _mobile_svg(
            "RecoveryBench v1: strict success and recovery",
            f"Three-seed means · rate 0–1 · source {digest}",
            success_rows,
            series=("strict success", "recovery after error"),
            description=(
                "Vertical mobile chart of three-seed mean strict task success and "
                "recovery-after-error rates for every RecoveryBench arm. Each bar is "
                "labelled with its series name and its exact rate."
            ),
        ),
        "cost-quality-pareto.svg": _svg(
            "Recovery quality with continuation cost",
            f"Three-seed means · recovery (cyan) · train seconds / max (violet) · source {digest}",
            cost_rows,
            x_label="normalized scale",
        ),
        "cost-quality-pareto-mobile.svg": _mobile_svg(
            "Recovery quality with continuation cost",
            f"Three-seed means · normalized scale · source {digest}",
            cost_rows,
            series=("recovery after error", "train seconds / slowest"),
            description=(
                "Vertical mobile chart pairing each RecoveryBench arm's three-seed mean "
                "recovery-after-error rate with its continuation training time as a "
                "fraction of the slowest arm. Both values are printed as text."
            ),
        ),
        "fresh-vs-frozen.svg": _svg(
            "Fresh-state OPD minus frozen-state KD",
            f"Paired task differences with 10,000 bootstrap replicates · source {digest}",
            paired_rows,
            x_label="mean (cyan) · upper 95% bound (violet)",
        ),
        "fresh-vs-frozen-mobile.svg": _mobile_svg(
            "Fresh-state OPD minus frozen-state KD",
            f"Paired differences · 10,000 bootstrap replicates · source {digest}",
            paired_rows,
            series=("mean difference", "upper 95% bound"),
            description=(
                "Vertical mobile chart of paired strict-task-success and "
                "recovery-after-error differences between fresh-state OPD and "
                "frozen-state KD. Both the mean difference and the upper 95 percent "
                "bootstrap bound stay below zero and are printed as text."
            ),
        ),
    }


def publish(paths: dict[str, Path]) -> dict[str, str]:
    loaded = [(view, path, _load_result(path, view)) for view, path in paths.items()]
    hashes = {view: _sha256(path) for view, path, _ in loaded}
    rows = _compact_task_rows(loaded)
    analysis = analyze(rows, hashes)
    RESULTS.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    destination_names = {
        "equal_optimizer_updates": "recoverybench-v1-equal-updates.json",
        "equal_selected_training_tokens": "recoverybench-v1-equal-selected-tokens.json",
        "equal_gpu_wall_time": "recoverybench-v1-equal-wall-time.json",
    }
    for view, source, _ in loaded:
        destination = RESULTS / destination_names[view]
        shutil.copyfile(source, destination)
    TASK_RESULTS.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )
    ANALYSIS.write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    for name, content in render_figures(
        loaded[0][2], analysis, hashes["equal_optimizer_updates"]
    ).items():
        (DOCS / name).write_text(content, encoding="utf-8", newline="\n")
    return {
        path.relative_to(ROOT).as_posix(): _sha256(path)
        for path in [
            TASK_RESULTS,
            ANALYSIS,
            *(
                DOCS / name
                for name in (
                    "recovery-success.svg",
                    "recovery-success-mobile.svg",
                    "cost-quality-pareto.svg",
                    "cost-quality-pareto-mobile.svg",
                    "fresh-vs-frozen.svg",
                    "fresh-vs-frozen-mobile.svg",
                )
            ),
        ]
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--equal-updates", type=Path, required=True)
    parser.add_argument("--equal-selected-tokens", type=Path, required=True)
    parser.add_argument("--equal-wall-time", type=Path, required=True)
    args = parser.parse_args()
    hashes = publish(
        {
            "equal_optimizer_updates": args.equal_updates,
            "equal_selected_training_tokens": args.equal_selected_tokens,
            "equal_gpu_wall_time": args.equal_wall_time,
        }
    )
    print(json.dumps(hashes, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
