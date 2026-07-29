"""Scientifically controlled benchmark harness with resolved-config provenance."""

from __future__ import annotations

import copy
import gc
import hashlib
import sys
import time
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from miniverl import __version__
from miniverl.cache.store import sha256_file
from miniverl.config.models import RunConfig, TrainingMode
from miniverl.errors import ConfigError
from miniverl.evaluation.schema import ArmResult, BenchmarkConfig, BenchmarkResult, finite_or_none
from miniverl.utils import gpu
from miniverl.utils.env import collect_environment
from miniverl.utils.logging import get_logger
from miniverl.utils.runs import canonical_json, read_jsonl, utc_now, write_json, write_text

__all__ = [
    "deep_merge",
    "structured_diff",
    "portable_payload",
    "resolve_benchmark_configs",
    "run_benchmark",
    "render_benchmark_markdown",
]

logger = get_logger("benchmark")
_HARNESS_DIFFERENCES = {"run.name", "run.seed", "run.run_id", "report.enabled"}
_MISSING = object()


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def structured_diff(reference: Any, candidate: Any, path: str = "") -> list[dict[str, Any]]:
    """Return deterministic leaf-level differences between JSON-compatible values."""
    if isinstance(reference, dict) and isinstance(candidate, dict):
        rows: list[dict[str, Any]] = []
        for key in sorted(set(reference) | set(candidate)):
            child = f"{path}.{key}" if path else str(key)
            rows.extend(
                structured_diff(
                    reference.get(key, _MISSING),
                    candidate.get(key, _MISSING),
                    child,
                )
            )
        return rows
    if reference == candidate:
        return []
    return [
        {
            "path": path,
            "common": None if reference is _MISSING else reference,
            "arm": None if candidate is _MISSING else candidate,
            "change": (
                "added"
                if reference is _MISSING
                else "removed"
                if candidate is _MISSING
                else "changed"
            ),
        }
    ]


def _path_allowed(path: str, patterns: set[str]) -> bool:
    for pattern in patterns:
        if pattern.endswith(".*"):
            prefix = pattern[:-2]
            if path == prefix or path.startswith(prefix + "."):
                return True
        elif path == pattern:
            return True
    return False


def _digest_payload(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def portable_payload(value: Any) -> Any:
    """Replace machine-local absolute paths before hashing or publishing provenance."""
    if isinstance(value, dict):
        return {key: portable_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [portable_payload(item) for item in value]
    if isinstance(value, str):
        windows = PureWindowsPath(value)
        posix = PurePosixPath(value)
        if windows.is_absolute() or posix.is_absolute():
            name = windows.name if windows.is_absolute() else posix.name
            return f"<local>/{name or 'path'}"
    return value


def _load_base(config: BenchmarkConfig) -> dict[str, Any]:
    if isinstance(config.base, dict):
        return RunConfig.model_validate(config.base).model_dump(mode="json")
    return RunConfig.from_yaml(config.base).model_dump(mode="json")


def _cold_payload(
    common: dict[str, Any],
    config: BenchmarkConfig,
    *,
    seed: int,
) -> dict[str, Any]:
    payload = deep_merge(common, config.cold_start_overrides)
    return deep_merge(
        payload,
        {
            "run": {
                "mode": TrainingMode.SFT.value,
                "seed": seed,
                "name": f"{config.name}-coldstart",
            },
            "train": {
                "cycles": config.cold_start_cycles,
                "sft_warmup_cycles": 0,
                "eval_every_cycles": 0,
                "save_every_cycles": 0,
            },
            "cache": {"reuse_across_policy_versions": False, "strict_policy_version": True},
            "report": {"enabled": False},
            "eval": {"enabled": False},
        },
    )


def _arm_payload(
    common: dict[str, Any],
    arm_overrides: dict[str, Any],
    *,
    seed: int,
    name: str,
) -> dict[str, Any]:
    return deep_merge(
        deep_merge(common, arm_overrides),
        {
            "run": {"seed": seed, "name": name},
            "report": {"enabled": False},
        },
    )


def resolve_benchmark_configs(
    config: BenchmarkConfig,
    *,
    seed: int | None = None,
) -> tuple[RunConfig, RunConfig, list[tuple[Any, RunConfig, list[dict[str, Any]]]]]:
    """Resolve and validate common, cold-start and arm configs before model loading."""
    selected_seed = config.seeds[0] if seed is None else seed
    base = _load_base(config)
    common = RunConfig.model_validate(deep_merge(base, config.common_overrides))
    common_dump = common.model_dump(mode="json")
    cold = RunConfig.model_validate(_cold_payload(common_dump, config, seed=selected_seed))
    allowed = set(config.allowed_differences) | _HARNESS_DIFFERENCES
    resolved: list[tuple[Any, RunConfig, list[dict[str, Any]]]] = []
    undeclared: list[str] = []

    for arm in config.arms:
        name = f"{config.name}-{arm.name}-s{selected_seed}"
        arm_config = RunConfig.model_validate(
            _arm_payload(common_dump, arm.overrides, seed=selected_seed, name=name)
        )
        diff = structured_diff(common_dump, arm_config.model_dump(mode="json"))
        bad = [row["path"] for row in diff if not _path_allowed(str(row["path"]), allowed)]
        if bad:
            undeclared.extend(f"{arm.name}: {path}" for path in bad)
        resolved.append((arm, arm_config, diff))

    if undeclared:
        preview = ", ".join(undeclared[:12])
        suffix = "" if len(undeclared) <= 12 else f" (+{len(undeclared) - 12} more)"
        raise ConfigError(
            "benchmark contains undeclared arm differences: " + preview + suffix,
            hint=(
                "move matched settings into common_overrides, or list intentional "
                "leaf paths under allowed_differences. Validation happens before "
                "any model is loaded."
            ),
        )
    return common, cold, resolved


def _checkpoint_digest(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        file_digest, size = sha256_file(path)
        digest.update(path.relative_to(directory).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_digest.encode("ascii"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _cold_start(
    run_config: RunConfig,
    output_dir: Path,
    *,
    local_files_only: bool = False,
    resume_existing: bool = False,
) -> tuple[Path | None, float]:
    if run_config.train.cycles <= 0:
        return None, 0.0
    from miniverl.trainer import OPDTrainer

    started = time.perf_counter()
    run_id = f"{run_config.run.name}-s{run_config.run.seed}"
    existing = output_dir / run_id
    with OPDTrainer.from_config(
        run_config,
        output_dir=None if resume_existing and existing.is_dir() else output_dir,
        run_id=None if resume_existing and existing.is_dir() else run_id,
        resume=existing if resume_existing and existing.is_dir() else None,
        local_files_only=local_files_only,
    ) as trainer:
        result = trainer.train()
        checkpoint = trainer.paths.checkpoints / "final"
        logger.info(
            "cold start seed %d: %d optimizer steps",
            run_config.run.seed,
            result.global_step,
        )
        return checkpoint, time.perf_counter() - started


def _isolate_next_trainer(stage: str) -> None:
    """Collect dead tensor owners and return cached CUDA blocks to the driver."""
    gc.collect()
    gpu.empty_cache()
    snapshot = gpu.snapshot()
    logger.debug(
        "%s teardown: allocated=%d reserved=%d",
        stage,
        snapshot.allocated_bytes,
        snapshot.reserved_bytes,
    )


def _training_accounting(trainer: Any, mode: TrainingMode) -> dict[str, Any]:
    """Sum cycle accounting across the complete run, never the final cycle only."""
    rows = [
        row
        for row in read_jsonl(trainer.paths.metrics)
        if str(row.get("phase", "")).endswith("_cycle")
    ]
    trajectories = 0
    generated = 0
    selected = 0
    eligible = 0
    for row in rows:
        rollouts = row.get("rollouts") or {}
        selection = row.get("selection") or {}
        trajectories += int(rollouts.get("rollouts") or 0)
        generated += int(rollouts.get("generated_tokens") or 0)
        selected += int(selection.get("selected_model_tokens") or 0)
        eligible += int(selection.get("total_model_tokens") or 0)

    teacher_positions = selected if mode is not TrainingMode.SFT else None
    return {
        "total_trajectories": trajectories,
        "generated_training_tokens_total": generated,
        "selected_training_tokens_total": selected,
        "model_generated_training_tokens_total": (generated if mode is TrainingMode.OPD else 0),
        "selected_position_ratio": (selected / eligible) if eligible else None,
        "teacher_queried_positions_total": teacher_positions,
        "teacher_queried_position_ratio": (
            (teacher_positions / eligible) if teacher_positions is not None and eligible else None
        ),
    }


def _peak_memory(trainer: Any) -> tuple[int | None, int | None]:
    allocated = 0
    reserved = 0
    seen = False
    for record in read_jsonl(trainer.paths.metrics):
        memory = record.get("memory") or {}
        if memory.get("cuda_available"):
            seen = True
            allocated = max(allocated, int(memory.get("peak_allocated_bytes") or 0))
            reserved = max(reserved, int(memory.get("peak_reserved_bytes") or 0))
    return (allocated, reserved) if seen else (None, None)


def _classified_config_diffs(
    *,
    config: BenchmarkConfig,
    common_config: dict[str, Any],
    declared_arm_config: dict[str, Any],
    runtime_arm_config: dict[str, Any],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Return legacy, scientific, runtime and harness config differences."""
    declared = structured_diff(common_config, declared_arm_config)
    allowed = set(config.allowed_differences)
    scientific = [
        {**row, "stage": "declared_vs_common"}
        for row in declared
        if _path_allowed(str(row["path"]), allowed)
    ]
    harness = [
        {**row, "stage": "declared_vs_common"}
        for row in declared
        if _path_allowed(str(row["path"]), _HARNESS_DIFFERENCES)
    ]
    runtime_all = structured_diff(declared_arm_config, runtime_arm_config)
    runtime = [
        {**row, "stage": "runtime_resolution"}
        for row in runtime_all
        if not _path_allowed(str(row["path"]), _HARNESS_DIFFERENCES)
    ]
    harness.extend(
        {**row, "stage": "runtime_resolution"}
        for row in runtime_all
        if _path_allowed(str(row["path"]), _HARNESS_DIFFERENCES)
    )
    return declared, scientific, runtime, harness


def _run_one_arm(
    *,
    config: BenchmarkConfig,
    target: Path,
    seed: int,
    arm: Any,
    run_config: RunConfig,
    common_dump: dict[str, Any],
    checkpoint: Path | None,
    local_files_only: bool = False,
    resume_existing: bool = False,
) -> ArmResult:
    """Run one arm inside a model-owning lifetime boundary."""
    from miniverl.trainer import OPDTrainer
    from miniverl.training.checkpoint import load_checkpoint

    declared_config = portable_payload(run_config.model_dump(mode="json"))
    arm_started = time.perf_counter()
    existing = target / run_config.run.name
    with OPDTrainer.from_config(
        run_config,
        output_dir=None if resume_existing and existing.is_dir() else target,
        run_id=None if resume_existing and existing.is_dir() else run_config.run.name,
        resume=existing if resume_existing and existing.is_dir() else None,
        local_files_only=local_files_only,
    ) as trainer:
        runtime_config = portable_payload(
            RunConfig.from_yaml(trainer.paths.config_resolved).model_dump(mode="json")
        )
        legacy_diff, scientific_diff, runtime_diff, harness_diff = _classified_config_diffs(
            config=config,
            common_config=common_dump,
            declared_arm_config=declared_config,
            runtime_arm_config=runtime_config,
        )
        if checkpoint is not None and checkpoint.is_dir():
            load_checkpoint(
                checkpoint,
                backend=trainer.student,
                optimizer=trainer.optimizer,
                device=trainer.student.device,
                include_optimizer=False,
                include_rng=False,
            )
            trainer.events.emit(
                "benchmark_cold_start_loaded",
                checkpoint_digest=_checkpoint_digest(checkpoint),
                note="weights only; optimizer state and RNG intentionally not restored",
            )

        train_started = time.perf_counter()
        train_result = trainer.train()
        train_seconds = time.perf_counter() - train_started
        eval_started = time.perf_counter()
        evaluation = trainer.evaluate(
            split=config.eval_split,
            tag=f"benchmark-{arm.name}",
        )
        evaluation_seconds = time.perf_counter() - eval_started
        wall_seconds = time.perf_counter() - arm_started
        cache_stats = trainer._cache.stats() if trainer._cache is not None else None
        accounting = _training_accounting(trainer, run_config.run.mode)
        manifest = trainer.build_manifest()
        objective = manifest["objective"]
        models = manifest["models"]
        memory = _peak_memory(trainer)
        cache_written = (
            getattr(trainer._cache, "bytes_written_total", None)
            if trainer._cache is not None
            else None
        )
        has_cuda = memory[0] is not None

        result = ArmResult(
            name=arm.name,
            description=arm.description,
            mode=run_config.run.mode.value,
            seed=seed,
            run_id=trainer.run_id,
            run_dir=str(trainer.paths.root),
            objective=str(objective["name"]),
            opd_freshness=objective.get("opd_freshness"),
            loss_mode=objective.get("loss_mode"),
            divergence=objective.get("divergence"),
            selector=objective.get("selector"),
            top_k=objective.get("top_k"),
            resolved_config_digest=_digest_payload(declared_config),
            structured_diff=legacy_diff,
            declared_config_digest=_digest_payload(declared_config),
            runtime_resolved_config_digest=_digest_payload(runtime_config),
            scientific_config_diff=scientific_diff,
            runtime_resolution_diff=runtime_diff,
            harness_config_diff=harness_diff,
            student_model_id=models["student"]["model_id"],
            student_model_revision=models["student"]["revision"],
            teacher_model_id=(
                models["teacher"]["model_id"] if models.get("teacher") is not None else None
            ),
            teacher_model_revision=(
                models["teacher"]["revision"] if models.get("teacher") is not None else None
            ),
            teacher_adapter=(
                models["teacher"].get("adapter") if models.get("teacher") is not None else None
            ),
            tokenizer_fingerprint=models["tokenizer_fingerprint"],
            teacher_context_mode=(
                models["teacher"]["context_mode"] if models.get("teacher") is not None else None
            ),
            optimizer_steps=train_result.global_step,
            policy_version=train_result.policy_version,
            **accounting,
            tasks=int(evaluation["tasks"]),
            success_rate=float(evaluation["success_rate"]),
            strict_task_success_rate=finite_or_none(evaluation.get("strict_task_success_rate")),
            lenient_diagnostic_success_rate=finite_or_none(
                evaluation.get("lenient_diagnostic_success_rate")
            ),
            avg_turns=float(evaluation["avg_turns"]),
            avg_tool_calls=float(evaluation["avg_tool_calls"]),
            **{
                key: int(evaluation[key]) if evaluation.get(key) is not None else None
                for key in (
                    "assistant_turns",
                    "emitted_tool_calls",
                    "parsed_tool_calls",
                    "tool_execution_successes",
                    "tool_execution_errors",
                    "unknown_tool_calls",
                    "parse_errors",
                    "repeated_call_terminations",
                    "final_answers_emitted",
                    "final_answers_format_valid",
                    "final_answers_verified",
                )
            },
            parse_valid_tool_call_rate=finite_or_none(evaluation.get("parse_valid_tool_call_rate")),
            tool_execution_success_rate=finite_or_none(
                evaluation.get("tool_execution_success_rate")
            ),
            tool_execution_error_rate=finite_or_none(evaluation.get("tool_execution_error_rate")),
            tool_call_count=int(evaluation["tool_call_count"]),
            valid_tool_call_rate=finite_or_none(evaluation.get("valid_tool_call_rate")),
            invalid_tool_call_rate=float(evaluation["invalid_tool_call_rate"]),
            final_answer_format_validity_rate=finite_or_none(
                evaluation.get("final_answer_format_validity_rate")
            ),
            protocol_token_accuracy=finite_or_none(evaluation.get("protocol_token_accuracy")),
            generated_tokens_per_task=float(evaluation["generated_tokens_per_task"]),
            tokens_per_solved_task=finite_or_none(evaluation["tokens_per_solved_task"]),
            cache_current_bytes=(cache_stats.actual_bytes if cache_stats else None),
            cache_bytes_written_total=cache_written,
            cache_compression_ratio=(cache_stats.compression_ratio if cache_stats else None),
            peak_allocated_bytes=memory[0],
            peak_reserved_bytes=memory[1],
            train_seconds=train_seconds,
            evaluation_seconds=evaluation_seconds,
            wall_seconds=wall_seconds,
            measurement_status={
                "train_time": "measured",
                "evaluation_time": "measured",
                "wall_time": "measured",
                "peak_vram": ("measured" if has_cuda else "not_run_no_cuda"),
                "cache": ("measured" if cache_stats is not None else "not_applicable"),
                **evaluation.get("policy_competence_measurement_status", {}),
            },
        )
        logger.info(
            "arm %-18s seed %d: success %.3f in %d steps (%.1fs)",
            arm.name,
            seed,
            float(evaluation["success_rate"]),
            train_result.global_step,
            wall_seconds,
        )
        return result


def run_benchmark(
    config: BenchmarkConfig,
    *,
    output_dir: str | Path | None = None,
    notes: str = "",
    invocation: list[str] | None = None,
    local_files_only: bool = False,
    resume: bool = False,
) -> BenchmarkResult:
    """Execute every preflight-validated arm and write a schema-v2 result.

    Existing arm directories are always collisions unless ``resume=True``.
    Resume attaches to their validated checkpoints and intentionally extends
    the same logs; it never treats stale files as a new benchmark run.
    """
    # Preflight every seed before creating a run directory or allocating a model.
    preflight = {seed: resolve_benchmark_configs(config, seed=seed) for seed in config.seeds}
    common_config = preflight[config.seeds[0]][0]
    common_dump = portable_payload(common_config.model_dump(mode="json"))
    common_digest = _digest_payload(common_dump)

    target = Path(output_dir or config.output_dir)
    target.mkdir(parents=True, exist_ok=True)
    env_info = collect_environment()
    results: list[ArmResult] = []
    cold_checkpoints: list[dict[str, Any]] = []

    for seed in config.seeds:
        _, cold_config, resolved_arms = preflight[seed]
        try:
            checkpoint, cold_seconds = _cold_start(
                cold_config,
                target,
                local_files_only=local_files_only,
                resume_existing=resume,
            )
        finally:
            _isolate_next_trainer(f"cold start seed {seed}")
        cold_checkpoints.append(
            {
                "seed": seed,
                "run_id": f"{cold_config.run.name}-s{seed}",
                "resolved_config_digest": _digest_payload(
                    portable_payload(cold_config.model_dump(mode="json"))
                ),
                "checkpoint_digest": (
                    _checkpoint_digest(checkpoint)
                    if checkpoint is not None and checkpoint.is_dir()
                    else None
                ),
                "train_seconds": cold_seconds,
            }
        )

        for arm, run_config, _diff in resolved_arms:
            try:
                results.append(
                    _run_one_arm(
                        config=config,
                        target=target,
                        seed=seed,
                        arm=arm,
                        run_config=run_config,
                        common_dump=common_dump,
                        checkpoint=checkpoint,
                        local_files_only=local_files_only,
                        resume_existing=resume,
                    )
                )
            finally:
                _isolate_next_trainer(f"arm {arm.name} seed {seed}")

    cold_dump = portable_payload(preflight[config.seeds[0]][1].model_dump(mode="json"))
    result = BenchmarkResult(
        miniverl_version=__version__,
        name=config.name,
        description=config.description,
        created_at=utc_now(),
        git_commit=env_info["git_commit"],
        invocation=portable_payload(list(invocation or sys.argv)),
        budget_axis=config.budget_axis,
        hardware={
            "gpu": env_info["gpu"],
            "os": env_info["os"],
            "cpu_count": env_info["cpu_count"],
        },
        software={
            "python": env_info["python_version"],
            "packages": env_info["packages"],
        },
        cold_start={
            "cycles": config.cold_start_cycles,
            "resolved_config": cold_dump,
            "resolved_config_digest": _digest_payload(cold_dump),
            "environment": cold_dump["environment"]["name"],
            "difficulty": cold_dump["environment"]["difficulty"],
            "checkpoints": cold_checkpoints,
        },
        common_resolved_config=common_dump,
        common_resolved_config_digest=common_digest,
        common_declared_config=common_dump,
        common_declared_config_digest=common_digest,
        controlled={
            "source": "common_declared_config",
            "digest": common_digest,
            "budget_axis": config.budget_axis,
            "allowed_differences": list(config.allowed_differences),
        },
        arms=results,
        notes=notes,
        seeds=list(config.seeds),
    )
    write_json(target / f"{config.name}.json", result.model_dump(mode="json"))
    write_text(target / f"{config.name}.md", render_benchmark_markdown(result))
    return result


def render_benchmark_markdown(result: BenchmarkResult) -> str:
    """Render a benchmark result without hiding single-seed or budget caveats."""
    gpu = result.hardware.get("gpu") or {}
    lines = [
        f"# Benchmark `{result.name}`",
        "",
        result.description or "",
        "",
        f"- schema v{result.schema_version} | miniVERL {result.miniverl_version} "
        f"| git `{result.git_commit or 'n/a'}`",
        f"- created {result.created_at}",
        f"- budget axis: `{result.budget_axis or 'legacy-unreported'}`",
        "- hardware: "
        + (
            f"{gpu.get('name')} ({gpu.get('total_memory_gib')} GiB), driver "
            f"{gpu.get('driver_version')}"
            if gpu.get("available")
            else "CPU only (no CUDA device visible)"
        ),
        f"- seeds: {result.seeds}"
        + ("  **single seed -- no significance claimed**" if len(result.seeds) == 1 else ""),
        "",
        "## Results",
        "",
        "| arm | objective | steps | success | selected positions | teacher queried | "
        "train s | eval s | peak allocated |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in result.arms:
        teacher = (
            "n/a"
            if arm.teacher_queried_positions_total is None
            else str(arm.teacher_queried_positions_total)
        )
        peak = "n/a" if arm.peak_allocated_bytes is None else str(arm.peak_allocated_bytes)
        lines.append(
            f"| {arm.name} | {arm.objective} | {arm.optimizer_steps} | "
            f"{arm.success_rate * 100:.1f}% | {arm.selected_training_tokens_total} | "
            f"{teacher} | {(arm.train_seconds or 0.0):.1f} | "
            f"{(arm.evaluation_seconds or 0.0):.1f} | {peak} |"
        )
    lines.extend(
        [
            "",
            "## Resolved controls",
            "",
            "The complete common declared configuration and separate scientific, "
            "runtime-resolution and harness-only diffs are stored in the JSON artifact. "
            "Undeclared scientific differences are rejected before any model is loaded.",
            "",
        ]
    )
    if result.notes:
        lines.extend(["## Notes", "", result.notes, ""])
    return "\n".join(lines)
