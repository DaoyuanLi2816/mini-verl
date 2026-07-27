"""Matched-budget benchmark harness.

What "matched" means here, concretely
------------------------------------
Every arm gets:

* the **same task splits** (same environment, difficulty and ``split_seed``),
* the **same initial weights** (one shared SFT cold start, loaded weights-only
  into each arm so no optimizer momentum leaks between arms),
* the **same number of optimizer steps** and the same effective batch size,
* the **same maximum trajectory length** and rollout bounds,
* the **same evaluation split, seed and temperature** (greedy).

Arms differ only in the keys they override, and those keys are recorded in the
result file. Quantities that cannot be matched by construction -- student
generated tokens, selected training tokens, teacher query ratio, wall clock --
are *measured and reported per arm* instead of being pretended away.
"""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any

from miniverl import __version__
from miniverl.config.models import RunConfig, TrainingMode
from miniverl.evaluation.schema import ArmResult, BenchmarkConfig, BenchmarkResult, finite_or_none
from miniverl.utils.env import collect_environment
from miniverl.utils.logging import get_logger
from miniverl.utils.runs import utc_now, write_json

__all__ = ["deep_merge", "run_benchmark", "render_benchmark_markdown"]

logger = get_logger("benchmark")


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _load_base(config: BenchmarkConfig) -> dict[str, Any]:
    if isinstance(config.base, dict):
        return copy.deepcopy(config.base)
    return RunConfig.from_yaml(config.base).model_dump(mode="json")


def _cold_start(
    base: dict[str, Any], config: BenchmarkConfig, seed: int, output_dir: Path
) -> tuple[Path | None, float | None]:
    """Run the shared SFT cold start and return its checkpoint directory."""
    if config.cold_start_cycles <= 0:
        return None, None
    from miniverl.trainer import OPDTrainer

    payload = deep_merge(
        base,
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
            # The cold start exists only to produce one shared checkpoint. Its own
            # evaluations are never read, and on a real model each one costs
            # minutes of generation, so they are switched off here.
            "eval": {"enabled": False},
        },
    )
    run_config = RunConfig.model_validate(payload)
    trainer = OPDTrainer.from_config(
        run_config, output_dir=output_dir, run_id=f"{config.name}-coldstart-s{seed}"
    )
    try:
        result = trainer.train()
        checkpoint = trainer.paths.checkpoints / "final"
        # The shared starting point is measured by the `cold-start-only` arm on
        # the benchmark's own split, not here.
        logger.info("cold start seed %d: %d optimizer steps", seed, result.global_step)
        return checkpoint, None
    finally:
        trainer.close()


def _arm_overrides(arm_overrides: dict[str, Any], seed: int, name: str) -> dict[str, Any]:
    forced = {
        "run": {"seed": seed, "name": name},
        "report": {"enabled": False},
    }
    return deep_merge(arm_overrides, forced)


def run_benchmark(
    config: BenchmarkConfig,
    *,
    output_dir: str | Path | None = None,
    notes: str = "",
) -> BenchmarkResult:
    """Execute every arm at every seed and return the measured result."""
    from miniverl.trainer import OPDTrainer
    from miniverl.training.checkpoint import load_checkpoint

    base = _load_base(config)
    target = Path(output_dir or config.output_dir)
    target.mkdir(parents=True, exist_ok=True)
    env_info = collect_environment()

    base_config = RunConfig.model_validate(base)
    controlled = {
        "environment": base_config.environment.name,
        "difficulty": base_config.environment.difficulty,
        "split_seed": base_config.environment.split_seed,
        "train_tasks": base_config.environment.train_tasks,
        "eval_tasks": base_config.environment.eval_tasks,
        "test_tasks": base_config.environment.test_tasks,
        "eval_split": config.eval_split,
        "eval_temperature": base_config.eval.temperature,
        "eval_seed": base_config.eval.seed,
        "max_trajectory_tokens": base_config.rollout.max_total_tokens,
        "max_turns": base_config.rollout.max_turns,
        "effective_batch_trajectories": base_config.train.gradient_accumulation_steps,
        "rollouts_per_cycle": base_config.train.rollouts_per_cycle,
        "optimizer": base_config.train.optimizer.value,
        "learning_rate": base_config.train.learning_rate,
        "lr_schedule": base_config.train.lr_schedule.value,
        "cold_start_cycles": config.cold_start_cycles,
        "cold_start_mode": "sft" if config.cold_start_cycles else "none",
        "shared_initial_checkpoint": bool(config.cold_start_cycles),
        "seeds": list(config.seeds),
        "arms_differ_only_in": {a.name: a.overrides for a in config.arms},
    }

    results: list[ArmResult] = []
    for seed in config.seeds:
        checkpoint, cold_baseline = _cold_start(base, config, seed, target)
        for arm in config.arms:
            run_name = f"{config.name}-{arm.name}-s{seed}"
            payload = deep_merge(base, _arm_overrides(arm.overrides, seed, run_name))
            run_config = RunConfig.model_validate(payload)
            trainer = OPDTrainer.from_config(run_config, output_dir=target, run_id=run_name)
            started = time.perf_counter()
            try:
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
                        checkpoint=str(checkpoint),
                        note="weights only; optimizer state and RNG intentionally not restored",
                    )
                train_result = trainer.train()
                evaluation = trainer.evaluate(split=config.eval_split, tag=f"benchmark-{arm.name}")
                elapsed = time.perf_counter() - started
                cache_stats = trainer._cache.stats() if trainer._cache is not None else None
                selection = _last_selection(trainer)
                objective = trainer.build_manifest()["objective"]
                memory = _peak_memory(trainer)
                results.append(
                    ArmResult(
                        name=arm.name,
                        description=arm.description,
                        mode=run_config.run.mode.value,
                        seed=seed,
                        run_id=trainer.run_id,
                        run_dir=str(trainer.paths.root),
                        loss_mode=run_config.loss.mode.value,
                        divergence=run_config.loss.divergence.value,
                        selector=run_config.selection.selector.value,
                        top_k=int(objective["top_k"]),
                        optimizer_steps=train_result.global_step,
                        policy_version=train_result.policy_version,
                        tasks=int(evaluation["tasks"]),
                        success_rate=float(evaluation["success_rate"]),
                        avg_turns=float(evaluation["avg_turns"]),
                        avg_tool_calls=float(evaluation["avg_tool_calls"]),
                        invalid_tool_call_rate=float(evaluation["invalid_tool_call_rate"]),
                        generated_tokens_per_task=float(evaluation["generated_tokens_per_task"]),
                        tokens_per_solved_task=finite_or_none(evaluation["tokens_per_solved_task"]),
                        selected_training_tokens=int(selection.get("selected_model_tokens", 0)),
                        teacher_queried_position_ratio=selection.get(
                            "teacher_queried_position_ratio"
                        ),
                        cache_bytes=(cache_stats.actual_bytes if cache_stats else None),
                        cache_compression_ratio=(
                            cache_stats.compression_ratio if cache_stats else None
                        ),
                        peak_allocated_bytes=memory[0],
                        peak_reserved_bytes=memory[1],
                        seconds=elapsed,
                        baseline_success_rate=cold_baseline,
                    )
                )
                logger.info(
                    "arm %-18s seed %d: success %.3f in %d steps (%.1fs)",
                    arm.name,
                    seed,
                    float(evaluation["success_rate"]),
                    train_result.global_step,
                    elapsed,
                )
            finally:
                trainer.close()

    result = BenchmarkResult(
        miniverl_version=__version__,
        name=config.name,
        description=config.description,
        created_at=utc_now(),
        git_commit=env_info["git_commit"],
        hardware={"gpu": env_info["gpu"], "os": env_info["os"], "cpu_count": env_info["cpu_count"]},
        software={"python": env_info["python_version"], "packages": env_info["packages"]},
        controlled=controlled,
        arms=results,
        notes=notes,
        seeds=list(config.seeds),
    )
    write_json(target / f"{config.name}.json", result.model_dump(mode="json"))
    (target / f"{config.name}.md").write_text(render_benchmark_markdown(result), encoding="utf-8")
    return result


def _last_selection(trainer: Any) -> dict[str, Any]:
    from miniverl.utils.runs import read_jsonl

    rows = [
        r
        for r in read_jsonl(trainer.paths.metrics)
        if str(r.get("phase", "")).endswith("_cycle") and r.get("selection")
    ]
    return dict(rows[-1]["selection"]) if rows else {}


def _peak_memory(trainer: Any) -> tuple[int | None, int | None]:
    from miniverl.utils.runs import read_jsonl

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


def render_benchmark_markdown(result: BenchmarkResult) -> str:
    """Render a benchmark result as a Markdown table with its controls."""
    gpu = result.hardware.get("gpu") or {}
    lines = [
        f"# Benchmark `{result.name}`",
        "",
        result.description or "",
        "",
        f"- miniVERL {result.miniverl_version} | git `{result.git_commit or 'n/a'}`",
        f"- created {result.created_at}",
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
        "| arm | mode | loss mode | steps | success | avg turns | invalid calls "
        "| gen tok/task | selected tokens | cache | seconds |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for arm in result.arms:
        cache = (
            f"{arm.cache_bytes / 1048576:.2f} MiB / {arm.cache_compression_ratio:.0f}x"
            if arm.cache_bytes
            else "-"
        )
        lines.append(
            f"| {arm.name} | {arm.mode} | {arm.loss_mode} | {arm.optimizer_steps} | "
            f"{arm.success_rate * 100:.1f}% | {arm.avg_turns:.2f} | "
            f"{arm.invalid_tool_call_rate * 100:.1f}% | {arm.generated_tokens_per_task:.1f} | "
            f"{arm.selected_training_tokens} | {cache} | {arm.seconds:.1f} |"
        )
    lines += [
        "",
        "## Aggregate",
        "",
        "| arm | seeds | mean success | min | max |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in result.aggregate():
        lines.append(
            f"| {row['name']} | {row['seeds']} | {row['success_rate_mean'] * 100:.1f}% | "
            f"{row['success_rate_min'] * 100:.1f}% | {row['success_rate_max'] * 100:.1f}% |"
        )
    lines += [
        "",
        "## What was held constant",
        "",
        "```json",
        __import__("json").dumps(result.controlled, indent=2, sort_keys=True),
        "```",
        "",
        "Arms differ **only** in the override keys listed above. Student generated tokens,",
        "selected training tokens, teacher query ratio and wall clock cannot be matched by",
        "construction, so they are measured and reported per arm rather than equalized.",
        "",
    ]
    if result.notes:
        lines += ["## Notes", "", result.notes, ""]
    return "\n".join(lines)
