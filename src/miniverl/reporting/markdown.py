"""Markdown and JSON run summaries.

The Markdown summary is what you paste into an issue or a PR; the JSON summary
is what a script consumes.  Both come from the same :class:`ReportData` as the
HTML report, so the three can never disagree.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from miniverl.reporting.data import ReportData

__all__ = ["render_markdown", "write_markdown", "render_summary_json"]


def _pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "n/a"


def _num(value: Any, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if number != number:
        return "n/a"
    return f"{number:.{digits}f}"


def render_markdown(data: ReportData) -> str:
    """Render a compact Markdown summary of a run."""
    manifest = data.manifest
    objective = manifest.get("objective") or {}
    gpu = manifest.get("gpu") or {}
    throughput = data.throughput()
    lines: list[str] = [
        f"# miniVERL run `{data.run_id}`",
        "",
        f"- mode: **{data.mode}**"
        + ("  (genuine on-policy distillation)" if data.is_on_policy else "  (not on-policy)"),
        f"- environment: `{(manifest.get('environment') or {}).get('name')}`"
        f" difficulty `{(manifest.get('environment') or {}).get('difficulty')}`",
        f"- objective: `{objective.get('divergence')}` / `{objective.get('loss_mode')}`"
        f" top_k={objective.get('top_k')} T={objective.get('temperature')}",
        f"- selector: `{objective.get('selector')}`",
        f"- seed: {manifest.get('seed')} | deterministic: {manifest.get('deterministic')}",
        f"- miniVERL {manifest.get('miniverl_version')} | git `{manifest.get('git_commit') or 'n/a'}`",
        "- device: "
        + (
            f"{gpu.get('name')} ({gpu.get('total_memory_gib')} GiB)"
            if gpu.get("available")
            else "CPU only (no CUDA device visible)"
        ),
        "",
        "## Results",
        "",
        "| checkpoint | step | tasks | success | avg turns | invalid calls | gen tokens/task |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in data.baseline_comparison():
        lines.append(
            f"| {row['label']} | {row['global_step']} | {row['tasks']} | "
            f"{_pct(row['success_rate'])} | {_num(row['avg_turns'], 2)} | "
            f"{_pct(row['invalid_tool_call_rate'])} | {_num(row['generated_tokens_per_task'], 1)} |"
        )
    if not data.baseline_comparison():
        lines.append("| _no evaluation recorded_ | | | | | | |")

    lines += [
        "",
        "## Throughput and memory",
        "",
        f"- optimizer steps: {throughput['optimizer_steps']}",
        f"- wall clock: {_num(throughput['wall_clock_seconds'], 2)} s",
        f"- rollout tokens/s (mean): {_num(throughput['rollout_tokens_per_second_mean'], 1)}",
        f"- training selected tokens/s (mean): "
        f"{_num(throughput['train_selected_tokens_per_second_mean'], 1)}",
        "- peak CUDA allocated: "
        + (
            f"{_num(throughput['peak_allocated_gib'], 3)} GiB"
            if throughput["peak_allocated_gib"] is not None
            else "not measured (no CUDA)"
        ),
        "- peak CUDA reserved: "
        + (
            f"{_num(throughput['peak_reserved_gib'], 3)} GiB"
            if throughput["peak_reserved_gib"] is not None
            else "not measured (no CUDA)"
        ),
    ]

    cache = data.cache_stats
    if cache and not cache.get("error"):
        lines += [
            "",
            "## Teacher-target cache",
            "",
            f"- trajectories: {cache['trajectories']} | selected positions: "
            f"{cache['selected_positions']}",
            f"- top-k {cache['top_k']} of vocab {cache['vocab_size']}",
            f"- on disk: {cache['actual_bytes']} B | dense fp16 reference: "
            f"{cache['theoretical_full_logit_bytes']} B",
            f"- compression: {_num(cache['compression_ratio'], 1)}x "
            f"({_num(cache['bytes_per_selected_position'], 1)} B per selected position)",
            f"- policy versions present: {cache['policy_versions']}",
            f"- checksum problems: {len(cache.get('problems') or [])}",
        ]

    failures = data.failure_counts()
    if failures:
        lines += ["", "## Failure taxonomy", ""]
        lines += [f"- `{name}`: {int(count)}" for name, count in failures]

    if data.benchmark:
        lines += [
            "",
            "## Matched-budget benchmark",
            "",
            "| arm | mode | steps | success | gen tokens/task | selected tokens | seconds |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for arm in data.benchmark.get("arms", []):
            lines.append(
                f"| {arm.get('name')} | {arm.get('mode')} | {arm.get('optimizer_steps')} | "
                f"{_pct(arm.get('success_rate'))} | "
                f"{_num(arm.get('generated_tokens_per_task'), 1)} | "
                f"{arm.get('selected_training_tokens')} | {_num(arm.get('seconds'), 1)} |"
            )

    lines += [
        "",
        "---",
        "",
        "Every number above was read from this run's own artifacts. Nothing is simulated;",
        "quantities that were not measured are printed as `n/a` with the reason.",
        "",
    ]
    return "\n".join(lines)


def write_markdown(data: ReportData, out: str | Path) -> Path:
    """Write :func:`render_markdown` to ``out``."""
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_markdown(data), encoding="utf-8")
    return target


def render_summary_json(data: ReportData) -> dict[str, Any]:
    """Machine-readable summary of a run."""
    return {
        "run_id": data.run_id,
        "mode": data.mode,
        "is_on_policy": data.is_on_policy,
        "manifest": data.manifest,
        "comparison": data.baseline_comparison(),
        "throughput": data.throughput(),
        "cache": data.cache_stats,
        "failure_categories": dict(data.failure_counts()),
        "termination_reasons": dict(data.termination_counts()),
        "selected_by_span_type": dict(data.selection_counts()),
        "benchmark": data.benchmark,
    }
