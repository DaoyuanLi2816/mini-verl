"""Export a run as a sanitized, schema-validated community benchmark result.

Sanitization is the point of this module.  A run directory contains absolute
paths and a machine description; a submission should contain neither the paths
nor anything identifying.  Only the fields listed in
:class:`~miniverl.evaluation.schema.BenchmarkResult` survive, ``run_dir`` is
reduced to the directory *name*, and the environment record is filtered down to
GPU model, VRAM, OS family and library versions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from miniverl import __version__
from miniverl.errors import ReportError
from miniverl.evaluation.schema import ArmResult, BenchmarkResult, finite_or_none
from miniverl.reporting.data import ReportData
from miniverl.utils.runs import utc_now, write_json

__all__ = ["export_run", "sanitize_hardware"]


def sanitize_hardware(environment: dict[str, Any]) -> dict[str, Any]:
    """Keep only non-identifying, reproducibility-relevant hardware facts."""
    gpu = environment.get("gpu") or {}
    return {
        "gpu_available": bool(gpu.get("available")),
        "gpu_name": gpu.get("name"),
        "gpu_total_memory_gib": gpu.get("total_memory_gib"),
        "gpu_capability": gpu.get("capability"),
        "gpu_driver_version": gpu.get("driver_version"),
        "gpu_count": gpu.get("device_count"),
        "os": environment.get("os"),
        "os_release": environment.get("os_release"),
        "machine": environment.get("machine"),
        "cpu_count": environment.get("cpu_count"),
    }


def export_run(
    run_dir: str | Path, *, out: str | Path | None = None, notes: str = ""
) -> tuple[dict[str, Any], Path]:
    """Build a submission-ready result file from one run directory."""
    data = ReportData.from_run(run_dir, max_trajectories=0, max_tokens=0)
    data.validate()
    manifest = data.manifest
    summary = data.summary
    final = summary.get("eval") or {}
    if not final.get("tasks"):
        raise ReportError(
            f"{data.run_dir} has no evaluation results to export",
            hint="run `miniverl eval --run <run-dir>` first, or train with eval.enabled: true",
        )

    objective = manifest.get("objective") or {}
    throughput = data.throughput()
    selection = data.selection_counts()
    cache = data.cache_stats or {}
    tokens_per_solved = finite_or_none(final.get("tokens_per_solved_task"))

    arm = ArmResult(
        name=str(manifest.get("run_name") or data.run_id),
        description=str((manifest.get("environment") or {}).get("name", "")),
        mode=data.mode,
        seed=int(manifest.get("seed") or 0),
        run_id=data.run_id,
        run_dir=Path(data.run_dir).name,
        objective=str(objective.get("name") or "legacy_unreported"),
        opd_freshness=objective.get("opd_freshness"),
        loss_mode=objective.get("loss_mode"),
        divergence=objective.get("divergence"),
        selector=objective.get("selector"),
        top_k=objective.get("top_k"),
        optimizer_steps=int(throughput["optimizer_steps"]),
        policy_version=int(summary.get("policy_version") or 0),
        tasks=int(final["tasks"]),
        success_rate=float(final["success_rate"]),
        strict_task_success_rate=finite_or_none(final.get("strict_task_success_rate")),
        lenient_diagnostic_success_rate=finite_or_none(
            final.get("lenient_diagnostic_success_rate")
        ),
        avg_turns=float(final["avg_turns"]),
        avg_tool_calls=float(final["avg_tool_calls"]),
        tool_call_count=(
            int(final["tool_call_count"]) if final.get("tool_call_count") is not None else None
        ),
        valid_tool_call_rate=finite_or_none(final.get("valid_tool_call_rate")),
        invalid_tool_call_rate=float(final["invalid_tool_call_rate"]),
        final_answer_format_validity_rate=finite_or_none(
            final.get("final_answer_format_validity_rate")
        ),
        protocol_token_accuracy=finite_or_none(final.get("protocol_token_accuracy")),
        generated_tokens_per_task=float(final["generated_tokens_per_task"]),
        tokens_per_solved_task=tokens_per_solved,
        selected_training_tokens_total=int(sum(count for _, count in selection)),
        teacher_queried_positions_total=(
            None if data.mode == "sft" else int(sum(count for _, count in selection))
        ),
        cache_current_bytes=cache.get("actual_bytes"),
        cache_compression_ratio=cache.get("compression_ratio"),
        peak_allocated_bytes=(
            int(throughput["peak_allocated_gib"] * 1024**3)
            if throughput["peak_allocated_gib"] is not None
            else None
        ),
        peak_reserved_bytes=(
            int(throughput["peak_reserved_gib"] * 1024**3)
            if throughput["peak_reserved_gib"] is not None
            else None
        ),
        wall_seconds=float(summary.get("duration_seconds") or 0.0),
        baseline_success_rate=(summary.get("baseline_eval") or {}).get("success_rate"),
        measurement_status={
            "wall_time": "measured",
            "peak_vram": ("measured" if throughput["cuda_available"] else "not_run_no_cuda"),
            "cache": "measured" if cache else "not_applicable",
        },
    )

    models = manifest.get("models") or {}
    result = BenchmarkResult(
        schema_version=1,
        miniverl_version=__version__,
        name=f"community-{data.run_id}",
        description=(
            f"Single-run community submission from {data.mode} on "
            f"{(manifest.get('environment') or {}).get('name')}"
        ),
        created_at=utc_now(),
        git_commit=manifest.get("git_commit"),
        hardware=sanitize_hardware(data.environment),
        software={
            "python": manifest.get("python_version"),
            "packages": manifest.get("packages"),
        },
        controlled={
            "environment": (manifest.get("environment") or {}).get("name"),
            "difficulty": (manifest.get("environment") or {}).get("difficulty"),
            "split_seed": (manifest.get("environment") or {}).get("split_seed"),
            "split_sizes": (manifest.get("environment") or {}).get("split_sizes"),
            "student": (models.get("student") or {}).get("model_id"),
            "student_revision": (models.get("student") or {}).get("revision"),
            "teacher": (models.get("teacher") or {}).get("model_id"),
            "teacher_revision": (models.get("teacher") or {}).get("revision"),
            "quantization": (models.get("student") or {}).get("quantization"),
            "memory_strategy": (manifest.get("memory") or {}).get("strategy"),
            "eval_temperature": final.get("temperature"),
            "eval_split": final.get("split"),
            "seeds": [manifest.get("seed")],
        },
        arms=[arm],
        notes=notes,
        seeds=[int(manifest.get("seed") or 0)],
    )
    payload = result.model_dump(mode="json")
    destination = Path(out) if out else Path(data.run_dir) / "benchmark-submission.json"
    write_json(destination, payload)
    return payload, destination
