"""Eval-only, preregistered teacher qualification for RecoveryBench."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from miniverl import __version__
from miniverl.agent.loop import RolloutRunner, RolloutStats
from miniverl.cache.store import sha256_file
from miniverl.config.models import RunConfig
from miniverl.environments.base import make_splits
from miniverl.environments.registry import make_environment
from miniverl.errors import ConfigError
from miniverl.evaluation.recovery import trajectory_recovery_metrics
from miniverl.trajectory.io import write_trajectories
from miniverl.utils import gpu
from miniverl.utils.env import collect_environment
from miniverl.utils.runs import JsonlWriter, utc_now, write_json

__all__ = [
    "DEFAULT_RECOVERY_TEACHER_GATE",
    "apply_teacher_gate",
    "evaluate_teacher_candidate",
    "validate_gate_split",
]

DEFAULT_RECOVERY_TEACHER_GATE = {
    "strict_task_success_rate": 0.80,
    "recovery_after_error_rate": 0.75,
    "parse_valid_tool_call_rate": 0.95,
    "tool_execution_success_rate": 0.70,
}


def validate_gate_split(split: str) -> str:
    """Teacher selection is deliberately incapable of reading final test tasks."""
    if split != "eval":
        raise ConfigError("RecoveryBench teacher qualification uses the eval split only")
    return split


def apply_teacher_gate(
    metrics: dict[str, Any],
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Apply every minimum threshold, treating missing/non-numeric data as failure."""
    gate = thresholds or DEFAULT_RECOVERY_TEACHER_GATE
    checks: dict[str, dict[str, Any]] = {}
    for name, minimum in gate.items():
        actual = metrics.get(name)
        numeric = float(actual) if isinstance(actual, (int, float)) else None
        checks[name] = {
            "operator": "greater_than_or_equal",
            "minimum": float(minimum),
            "actual": numeric,
            "passed": numeric is not None and numeric >= minimum,
        }
    return {"passed": all(check["passed"] for check in checks.values()), "checks": checks}


def evaluate_teacher_candidate(
    config: RunConfig,
    *,
    candidate_id: str,
    out: str | Path,
    tasks: int | None = None,
    split: str = "eval",
    thresholds: dict[str, float] | None = None,
    local_files_only: bool = False,
) -> dict[str, Any]:
    """Evaluate one configured frozen teacher as a tool policy on eval only."""
    validate_gate_split(split)
    count = tasks or config.environment.eval_tasks
    if count < 1 or count > config.environment.eval_tasks:
        raise ConfigError(
            f"teacher qualification tasks must be within 1..{config.environment.eval_tasks}"
        )
    destination = Path(out).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    trajectories_path = destination / "trajectories.jsonl"
    task_results_path = destination / "task-results.jsonl"

    environment = make_environment(config.environment.name, **config.environment.params)
    teacher = None
    started = time.perf_counter()
    try:
        task_splits = make_splits(
            environment,
            counts={"train": 0, "eval": count, "test": 0},
            seed=config.environment.split_seed,
            difficulty=config.environment.difficulty,
        )
        from miniverl.models.factory import build_teacher, build_tokenizer, resolve_device

        device = resolve_device(config.models)
        tokenizer = build_tokenizer(config, local_files_only=local_files_only)
        teacher = build_teacher(
            config,
            tokenizer,
            device=device,
            local_files_only=local_files_only,
        )
        teacher.set_train(False)
        runner = RolloutRunner(
            backend=teacher,
            environment=environment,
            config=config.rollout,
        )
        stats = RolloutStats()
        trajectories = []
        gpu.reset_peak_stats()
        for offset, task in enumerate(task_splits["eval"]):
            trajectory = runner.rollout(
                task,
                policy_version=0,
                seed=config.eval.seed + offset,
                temperature=0.0,
                max_turns=config.eval.max_turns,
                trajectory_id=f"{task.task_id}:teacher-gate:{candidate_id}",
            )
            trajectories.append(trajectory)
            stats.observe(trajectory)
        elapsed = time.perf_counter() - started
        write_trajectories(trajectories_path, trajectories)
        task_writer = JsonlWriter(task_results_path)
        for trajectory in trajectories:
            recovery = trajectory_recovery_metrics(trajectory)
            task_writer.write(
                {
                    "candidate_id": candidate_id,
                    "task_id": trajectory.task_id,
                    "strict_success": bool(
                        trajectory.verification and trajectory.verification.solved
                    ),
                    **recovery.__dict__,
                }
            )
        trajectory_sha, trajectory_bytes = sha256_file(trajectories_path)
        task_sha, task_bytes = sha256_file(task_results_path)
        metrics = stats.to_dict()
        gate = apply_teacher_gate(metrics, thresholds)
        environment_info = collect_environment()
        result = {
            "schema_version": 1,
            "status": "completed",
            "candidate_id": candidate_id,
            "created_at": utc_now(),
            "miniverl_version": __version__,
            "git_commit": environment_info["git_commit"],
            "split": "eval",
            "tasks": count,
            "test_tasks_generated": 0,
            "student_policy_used": False,
            "teacher": {
                "model_id": config.models.teacher.model_id,
                "revision": config.models.teacher.revision,
                "adapter": getattr(teacher, "adapter_provenance", None),
            },
            "metrics": metrics,
            "gate": gate,
            "seconds": elapsed,
            "memory": gpu.snapshot().to_dict(),
            "artifacts": {
                "trajectories.jsonl": {
                    "sha256": trajectory_sha,
                    "bytes": trajectory_bytes,
                },
                "task-results.jsonl": {"sha256": task_sha, "bytes": task_bytes},
            },
        }
        write_json(destination / "result.json", result)
        return result
    finally:
        if teacher is not None:
            teacher.release()
        closer = getattr(environment, "close", None)
        if callable(closer):
            closer()
        gpu.empty_cache()
