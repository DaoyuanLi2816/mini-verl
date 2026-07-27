"""Self-contained HTML report renderer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from miniverl import __version__
from miniverl.reporting.charts import bar_chart, line_chart, token_strip
from miniverl.reporting.data import ReportData
from miniverl.utils.runs import utc_now

__all__ = ["render_html", "write_report"]

_TEMPLATE_DIR = Path(__file__).parent / "templates"


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
    if number != number:  # NaN
        return "n/a"
    return f"{number:.{digits}f}"


def _mem(value: Any) -> str:
    if value is None:
        return "n/a (no CUDA)"
    return f"{float(value):.3f} GiB"


def _bytes_h(value: Any) -> str:
    try:
        size = float(value)
    except (TypeError, ValueError):
        return "n/a"
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(size) < 1024.0 or unit == "TiB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024.0
    return f"{size:.1f} TiB"  # pragma: no cover


def _event_detail(event: dict[str, Any]) -> str:
    skip = {"ts", "event"}
    parts = []
    for key, value in event.items():
        if key in skip:
            continue
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)[:160]
        parts.append(f"{key}={value}")
    return " ".join(parts)


def _tiles(data: ReportData) -> list[dict[str, str]]:
    summary = data.summary
    final = summary.get("eval") or {}
    baseline = summary.get("baseline_eval") or {}
    throughput = data.throughput()
    manifest = data.manifest
    objective = manifest.get("objective") or {}
    memory = manifest.get("memory") or {}
    delta = None
    if final.get("success_rate") is not None and baseline.get("success_rate") is not None:
        delta = float(final["success_rate"]) - float(baseline["success_rate"])
    tiles = [
        {
            "label": "task success",
            "value": _pct(final.get("success_rate")),
            "note": (
                f"{_pct(baseline.get('success_rate'))} before training"
                + (f" ({delta * 100:+.1f} pts)" if delta is not None else "")
            ),
        },
        {
            "label": "objective",
            "value": str(objective.get("divergence", "?")),
            "note": f"{objective.get('loss_mode', '?')}, k={objective.get('top_k', '?')}",
        },
        {
            "label": "optimizer steps",
            "value": str(throughput["optimizer_steps"]),
            "note": f"policy version {summary.get('policy_version', '?')}",
        },
        {
            "label": "peak VRAM (reserved)",
            "value": _mem(throughput["peak_reserved_gib"]),
            "note": (
                f"allocated {_mem(throughput['peak_allocated_gib'])}"
                if throughput["cuda_available"]
                else "no CUDA device in this run"
            ),
        },
        {
            "label": "memory strategy",
            "value": str(memory.get("strategy", "?")),
            "note": f"chunk {memory.get('projection_chunk_size', '?')}",
        },
        {
            "label": "wall clock",
            "value": f"{_num(throughput['wall_clock_seconds'], 1)} s",
            "note": f"{_num(throughput['rollout_tokens_per_second_mean'], 0)} rollout tok/s",
        },
    ]
    cache = data.cache_stats
    if cache and not cache.get("error"):
        tiles.append(
            {
                "label": "cache compression",
                "value": f"{_num(cache.get('compression_ratio'), 1)}x",
                "note": (
                    f"{_bytes_h(cache.get('actual_bytes'))} vs "
                    f"{_bytes_h(cache.get('theoretical_full_logit_bytes'))} dense fp16"
                ),
            }
        )
    return tiles


def _manifest_rows(data: ReportData) -> list[tuple[str, str]]:
    manifest = data.manifest
    gpu = manifest.get("gpu") or {}
    packages = manifest.get("packages") or {}
    models = manifest.get("models") or {}
    student = models.get("student") or {}
    teacher = models.get("teacher") or {}
    rows: list[tuple[str, str]] = [
        ("miniVERL version", str(manifest.get("miniverl_version"))),
        ("git commit", str(manifest.get("git_commit") or "not a git checkout")),
        ("created", str(manifest.get("created_at"))),
        ("mode", str(manifest.get("mode"))),
        ("seed", str(manifest.get("seed"))),
        ("deterministic", str(manifest.get("deterministic"))),
        ("python", str(manifest.get("python_version"))),
        ("os", f"{manifest.get('os')} {manifest.get('os_release')} ({manifest.get('platform')})"),
        (
            "gpu",
            (
                f"{gpu.get('name')} | {gpu.get('total_memory_gib')} GiB | "
                f"capability {gpu.get('capability')} | driver {gpu.get('driver_version')}"
            )
            if gpu.get("available")
            else f"not available ({gpu.get('reason')})",
        ),
        ("torch", str(packages.get("torch"))),
        ("torch cuda", str(gpu.get("torch_cuda_version"))),
        ("transformers", str(packages.get("transformers"))),
        ("peft", str(packages.get("peft"))),
        ("bitsandbytes", str(packages.get("bitsandbytes") or "not installed")),
        (
            "student",
            f"{student.get('model_id')} @ {student.get('revision') or 'unpinned'} | "
            f"{student.get('precision')} | quant {student.get('quantization')} | "
            f"{(student.get('capabilities') or {}).get('num_trainable_parameters')} trainable params",
        ),
        (
            "teacher",
            (
                f"{teacher.get('model_id')} @ {teacher.get('revision') or 'unpinned'} | "
                f"{teacher.get('precision')} | quant {teacher.get('quantization')} | "
                f"context {teacher.get('context_mode')}"
            )
            if teacher
            else "none (SFT run)",
        ),
        ("tokenizer fingerprint", str(models.get("tokenizer_fingerprint", ""))[:32] + "..."),
        ("tokenizer vocab", str(models.get("tokenizer_vocab_size"))),
        ("measurement status", json.dumps(manifest.get("measurement_status") or {})),
    ]
    return rows


def _benchmark_controls(benchmark: dict[str, Any] | None) -> str:
    if not benchmark:
        return ""
    controls = benchmark.get("controlled") or {}
    return json.dumps(controls, indent=2, sort_keys=True)


def render_html(data: ReportData) -> str:
    """Render the report to a single self-contained HTML string."""
    data.validate()
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=select_autoescape(["html", "j2"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template("report.html.j2")
    token_views = {
        traj_id: token_strip(records)
        for traj_id, records in data.token_analysis.items()
    }
    return template.render(
        data=data,
        manifest=data.manifest,
        version=__version__,
        generated_at=utc_now(),
        tiles=_tiles(data),
        comparison=data.baseline_comparison(),
        throughput=data.throughput(),
        cache=data.cache_stats,
        benchmark=data.benchmark,
        benchmark_controls=_benchmark_controls(data.benchmark),
        manifest_rows=_manifest_rows(data),
        token_views=token_views,
        loss_chart=line_chart(
            data.loss_series(), x_label="optimizer step", y_label="loss", title="objective"
        ),
        eval_chart=(
            line_chart(
                data.eval_series(),
                x_label="optimizer step",
                y_label="success rate",
                title="task success rate",
            )
            if data.eval_series()
            else ""
        ),
        selection_chart=bar_chart(data.selection_counts(), title="selected tokens by span type"),
        failure_chart=bar_chart(data.failure_counts(), title="failure categories"),
        termination_chart=bar_chart(data.termination_counts(), title="termination reasons"),
        pct=_pct,
        num=_num,
        mem=_mem,
        bytes_h=_bytes_h,
        event_detail=_event_detail,
    )


def write_report(
    run_dir: str | Path,
    out: str | Path | None = None,
    *,
    max_trajectories: int = 5,
    max_tokens: int = 400,
) -> Path:
    """Build a report for ``run_dir`` and write it to ``out``."""
    data = ReportData.from_run(
        run_dir, max_trajectories=max_trajectories, max_tokens=max_tokens
    )
    target = Path(out) if out is not None else Path(run_dir) / "report.html"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_html(data), encoding="utf-8")
    return target
