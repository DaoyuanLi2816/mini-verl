#!/usr/bin/env python3
"""Run the preregistered v0.4 consumer-runtime benchmark on one CUDA GPU."""

from __future__ import annotations

import argparse
import hashlib
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from miniverl.config import RunConfig
from miniverl.models.shared import PolicyRole
from miniverl.training.batching import (
    build_padded_trajectory_batch,
    deterministic_length_batches,
)
from miniverl.training.optim import build_optimizer
from miniverl.utils import gpu
from miniverl.utils.env import collect_environment
from miniverl.utils.runs import canonical_json, write_json

REPO_ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION = REPO_ROOT / "benchmarks/preregistration/consumer-runtime-v1.yaml"
FROZEN_CALCULATOR = REPO_ROOT / "benchmarks/results/gpu-calc-hard-equal-update-v2.json"
FROZEN_CALCULATOR_SHA256 = "53fc1d4d5b7adee09618d77ad62d4086ba56b78569832d6fc7c3bcd5c2695bbc"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _sync() -> None:
    torch.cuda.synchronize()


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def _config(*, runtime: str, batch_size: int | str, output_dir: Path) -> RunConfig:
    prereg = yaml.safe_load(PREREGISTRATION.read_text(encoding="utf-8"))
    model = prereg["model"]
    workload = prereg["workload"]
    base = {
        "model_id": model["base_id"],
        "revision": model["base_revision"],
        "tokenizer_revision": model["tokenizer_revision"],
        "dtype": model["dtype"],
        "quantization": model["quantization"],
        "attn_implementation": "eager",
    }
    mapping: dict[str, Any] = {
        "schema_version": 1,
        "run": {
            "name": f"consumer-runtime-v1-{runtime}-batch{batch_size}",
            "mode": "opd",
            "seed": 20260801,
            "output_dir": str(output_dir),
            "deterministic": True,
            "tags": ["consumer-runtime-v1", "systems-benchmark"],
        },
        "models": {
            "backend": "hf",
            "runtime": runtime,
            "device": "cuda",
            "student": {
                **base,
                "gradient_checkpointing": True,
                "prepare_kbit_training": False,
                "lora": {
                    "enabled": True,
                    "r": model["student_adapter"]["r"],
                    "alpha": model["student_adapter"]["alpha"],
                    "dropout": model["student_adapter"]["dropout"],
                },
            },
            "teacher": {
                **base,
                "adapter": {
                    "path": model["frozen_teacher_adapter"]["repo"],
                    "source": "hub",
                    "revision": model["frozen_teacher_adapter"]["revision"],
                    "base_model_revision": model["base_revision"],
                },
            },
        },
        "environment": {
            "name": workload["environment"],
            "params": {"protocol_version": workload["protocol_version"]},
            "train_tasks": workload["ordered_tasks"],
            "eval_tasks": 1,
            "test_tasks": 0,
            "split_seed": workload["split_seed"],
            "difficulty": workload["difficulty"],
        },
        "rollout": {
            "max_turns": 5,
            "max_new_tokens_per_turn": 64,
            "max_total_tokens": workload["max_total_tokens"],
            "temperature": 0.0,
        },
        "selection": {"selector": workload["selector"]},
        "loss": {
            "mode": workload["objective"]["mode"],
            "divergence": workload["objective"]["divergence"],
            "top_k": workload["objective"]["top_k"],
            "temperature": workload["objective"]["temperature"],
            "chunk_size": workload["objective"]["chunk_size"],
        },
        "train": {
            "cycles": 1,
            "rollouts_per_cycle": workload["ordered_tasks"],
            "gradient_accumulation_steps": workload["ordered_tasks"],
            "trajectory_batch_size": batch_size,
            "length_bucketing": True,
            "learning_rate": 1.0e-4,
            "weight_decay": 0.0,
            "max_grad_norm": 1.0,
            "optimizer": "adamw8bit",
        },
        "memory": {"strategy": "resident", "oom_retries": 0},
        "cache": {"entries_per_shard": workload["ordered_tasks"]},
        "eval": {"enabled": False},
        "report": {"enabled": False},
    }
    return RunConfig.from_mapping(mapping)


def _trajectory_digest(samples: list[Any]) -> str:
    payload = [
        {
            "task_id": sample.trajectory.task_id,
            "token_ids": sample.trajectory.token_ids,
            "positions": sample.alignment.student_prediction_positions,
            "target_ids": sample.alignment.target_token_ids,
            "weights": sample.alignment.token_weights,
        }
        for sample in samples
    ]
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _teacher_target_digest(samples: list[Any]) -> str:
    digest = hashlib.sha256()
    for sample in samples:
        provider = sample.teacher.provider
        for name in ("topk_indices", "topk_log_probs", "tail_log_prob"):
            tensor = getattr(provider, name).detach().to("cpu").contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _score_samples(trainer: Any, base_samples: list[Any]) -> list[Any]:
    if trainer.scorer is None:
        raise RuntimeError("consumer runtime benchmark requires a teacher scorer")
    for sample in base_samples:
        sample.teacher = trainer.scorer.score(
            student=sample.trajectory,
            alignment=sample.alignment,
        )
    return base_samples


def _padding_tokens(samples: list[Any], batch_size: int) -> int:
    lengths = [len(sample.trajectory.token_ids) for sample in samples]
    total = 0
    for indices in deterministic_length_batches(lengths, batch_size=batch_size):
        bucket = [lengths[index] for index in indices]
        total += max(bucket) * len(bucket) - sum(bucket)
    return total


def _student_forward_seconds(trainer: Any, samples: list[Any], batch_size: int) -> float:
    indices = deterministic_length_batches(
        [len(sample.trajectory.token_ids) for sample in samples], batch_size=batch_size
    )
    _sync()
    started = time.perf_counter()
    for group in indices:
        rows = [samples[index] for index in group]
        batch = build_padded_trajectory_batch(
            token_ids=[sample.trajectory.token_ids for sample in rows],
            selected_positions=[sample.alignment.student_prediction_positions for sample in rows],
            pad_token_id=trainer.tokenizer.pad_token_id,
            device=trainer.student.device,
        )
        hidden = trainer.student.hidden_states_at_batch(batch, with_grad=False)
        del hidden
    _sync()
    return time.perf_counter() - started


def _switch_microseconds(trainer: Any) -> float | None:
    controller = getattr(trainer.student, "controller", None)
    if controller is None:
        return None
    iterations = 200
    _sync()
    started = time.perf_counter()
    for _ in range(iterations):
        with controller.activate(PolicyRole.TEACHER):
            pass
    _sync()
    return (time.perf_counter() - started) * 1.0e6 / iterations


def _reset_student(trainer: Any, state: dict[str, torch.Tensor]) -> None:
    trainer.student.load_trainable_state_dict(state)
    trainer.optimizer = build_optimizer(
        trainer.student.trainable_parameters(),
        trainer.config.train,
    )


def _one_update(trainer: Any, samples: list[Any]) -> dict[str, Any]:
    _sync()
    update_started = time.perf_counter()
    metrics = trainer._compute_group_gradients(samples, trainer.config.loss.chunk_size)
    commit = trainer._commit_update()
    _sync()
    return {
        **metrics,
        **commit,
        "update_seconds": time.perf_counter() - update_started,
    }


def _profile_update(
    trainer: Any, samples: list[Any], state: dict[str, torch.Tensor]
) -> list[dict[str, Any]]:
    _reset_student(trainer, state)
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(
        activities=activities,
        profile_memory=True,
        record_shapes=True,
    ) as profile:
        _one_update(trainer, samples)
    rows = []
    for event in sorted(
        profile.key_averages(),
        key=lambda item: float(getattr(item, "self_device_time_total", 0.0)),
        reverse=True,
    )[:20]:
        rows.append(
            {
                "name": event.key,
                "calls": int(event.count),
                "self_cpu_time_us": float(event.self_cpu_time_total),
                "self_device_time_us": float(getattr(event, "self_device_time_total", 0.0)),
                "self_cpu_memory_bytes": int(event.self_cpu_memory_usage),
                "self_device_memory_bytes": int(getattr(event, "self_device_memory_usage", 0)),
            }
        )
    return rows


def _run_cell(
    *,
    runtime: str,
    batch_size: int | str,
    work_dir: Path,
    local_files_only: bool,
    warmups: int,
    repetitions: int,
    profile: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]] | None]:
    from miniverl.trainer import OPDTrainer

    config = _config(runtime=runtime, batch_size=batch_size, output_dir=work_dir)
    run_id = f"consumer-runtime-v1-{runtime}-batch{batch_size}"
    trainer = OPDTrainer.from_config(
        config,
        run_id=run_id,
        overwrite=True,
        local_files_only=local_files_only,
    )
    try:
        tasks = trainer.splits["train"][: config.train.rollouts_per_cycle]
        trajectories, _ = trainer._collect(tasks, oracle=True)
        base_samples = trainer._build_samples_ce_only(trajectories)
        trajectory_digest = _trajectory_digest(base_samples)
        initial_state = trainer.student.trainable_state_dict()
        physical_batch_size = (
            len(base_samples)
            if config.train.trajectory_batch_size == "auto"
            else int(config.train.trajectory_batch_size)
        )

        for _ in range(warmups):
            _reset_student(trainer, initial_state)
            _score_samples(trainer, base_samples)
            _one_update(trainer, base_samples)

        measurements: list[dict[str, Any]] = []
        target_digests: list[str] = []
        for _ in range(repetitions):
            _reset_student(trainer, initial_state)
            forward_seconds = _student_forward_seconds(trainer, base_samples, physical_batch_size)
            gpu.reset_peak_stats()
            _sync()
            e2e_started = time.perf_counter()
            teacher_started = time.perf_counter()
            samples = _score_samples(trainer, base_samples)
            _sync()
            teacher_seconds = time.perf_counter() - teacher_started
            target_digests.append(_teacher_target_digest(samples))
            update = _one_update(trainer, samples)
            e2e_seconds = time.perf_counter() - e2e_started
            memory = gpu.snapshot()
            measurement = {
                **update,
                "student_forward_seconds": forward_seconds,
                "teacher_scoring_seconds": teacher_seconds,
                "end_to_end_seconds": e2e_seconds,
                "peak_allocated_bytes": memory.peak_allocated_bytes,
                "peak_reserved_bytes": memory.peak_reserved_bytes,
            }
            measurements.append(measurement)

        positions = int(measurements[0]["selected_positions"])
        e2e_times = [float(row["end_to_end_seconds"]) for row in measurements]
        update_times = [float(row["update_seconds"]) for row in measurements]
        teacher_times = [float(row["teacher_scoring_seconds"]) for row in measurements]
        forward_times = [float(row["student_forward_seconds"]) for row in measurements]
        peak_allocated = max(int(row["peak_allocated_bytes"]) for row in measurements)
        peak_reserved = max(int(row["peak_reserved_bytes"]) for row in measurements)
        result = {
            "runtime": runtime,
            "batch_size": batch_size,
            "status": "completed",
            "trajectories": len(base_samples),
            "selected_positions": positions,
            "trajectory_digest": trajectory_digest,
            "teacher_target_digest": target_digests[0],
            "teacher_target_digests_identical": len(set(target_digests)) == 1,
            "padding_tokens": _padding_tokens(base_samples, physical_batch_size),
            "physical_batches": len(
                deterministic_length_batches(
                    [len(sample.trajectory.token_ids) for sample in base_samples],
                    batch_size=physical_batch_size,
                )
            ),
            "student_forward_seconds_median": _median(forward_times),
            "teacher_scoring_seconds_median": _median(teacher_times),
            "update_seconds_median": _median(update_times),
            "end_to_end_seconds_median": _median(e2e_times),
            "end_to_end_seconds_min": min(e2e_times),
            "trajectories_per_second": len(base_samples) / _median(e2e_times),
            "selected_positions_per_second": positions / _median(e2e_times),
            "student_trajectories_per_second": len(base_samples) / _median(forward_times),
            "teacher_positions_per_second": positions / _median(teacher_times),
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "adapter_switch_microseconds": _switch_microseconds(trainer),
            "loss_values": [float(row["loss"]) for row in measurements],
            "measurements": measurements,
        }
        profile_rows = _profile_update(trainer, base_samples, initial_state) if profile else None
        return result, profile_rows
    finally:
        trainer.close()
        gpu.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "benchmarks/results/consumer-runtime-v1.json",
    )
    parser.add_argument(
        "--profiler-output",
        type=Path,
        default=REPO_ROOT / "benchmarks/results/consumer-runtime-v1-profiler.json",
    )
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("consumer-runtime-v1 requires one CUDA GPU")
    if _sha256(FROZEN_CALCULATOR) != FROZEN_CALCULATOR_SHA256:
        raise SystemExit("immutable calculator benchmark hash changed")
    prereg = yaml.safe_load(PREREGISTRATION.read_text(encoding="utf-8"))
    repetitions = 1 if args.quick else int(prereg["measurement"]["measured_updates"])
    warmups = 0 if args.quick else int(prereg["measurement"]["warmup_updates"])
    cells: list[dict[str, Any]] = []
    profiler: dict[str, Any] = {}
    for runtime in ("dual_model", "shared_backbone"):
        for batch_size in (1, 2, 4, "auto"):
            try:
                cell, profile_rows = _run_cell(
                    runtime=runtime,
                    batch_size=batch_size,
                    work_dir=args.work_dir,
                    local_files_only=args.offline,
                    warmups=warmups,
                    repetitions=repetitions,
                    profile=runtime == "shared_backbone" and batch_size == "auto",
                )
                cells.append(cell)
                if profile_rows is not None:
                    profiler["shared_backbone_auto"] = profile_rows
                print(
                    f"{runtime} batch={batch_size}: "
                    f"{cell['trajectories_per_second']:.3f} trajectories/s, "
                    f"{cell['peak_reserved_bytes'] / 2**30:.3f} GiB reserved",
                    flush=True,
                )
            except (RuntimeError, MemoryError) as exc:
                if not gpu.is_oom_error(exc):
                    raise
                cells.append(
                    {
                        "runtime": runtime,
                        "batch_size": batch_size,
                        "status": "oom",
                        "error": str(exc),
                    }
                )
                gpu.empty_cache()

    cells.append(
        {
            "runtime": "dual_model_swap",
            "batch_size": "auto",
            "status": "not_run",
            "reason": "NF4 models are device-pinned; swap is explicitly unsupported",
        }
    )
    trajectory_digests = {
        str(cell.get("trajectory_digest")) for cell in cells if cell.get("status") == "completed"
    }
    target_digests = {
        str(cell.get("teacher_target_digest"))
        for cell in cells
        if cell.get("status") == "completed"
    }
    payload = {
        "schema_version": 1,
        "name": "consumer-runtime-v1",
        "measurement_status": "quick_diagnostic" if args.quick else "measured_final",
        "preregistration_sha256": _sha256(PREREGISTRATION),
        "code_commit": _git_commit(),
        "frozen_calculator_sha256": _sha256(FROZEN_CALCULATOR),
        "environment": collect_environment(),
        "workload_invariants": {
            "trajectory_digests_identical": len(trajectory_digests) == 1,
            "teacher_target_digests_identical": len(target_digests) == 1,
            "trajectory_digest": next(iter(trajectory_digests), None),
            "teacher_target_digest": next(iter(target_digests), None),
        },
        "cells": cells,
        "larger_model_diagnostic": [
            {
                "size": "4B",
                "status": "not_run",
                "reason": "no preregistered compatible frozen teacher adapter",
            },
            {
                "size": "7B",
                "status": "not_run",
                "reason": "no preregistered compatible frozen teacher adapter",
            },
        ],
    }
    if not args.quick and (len(trajectory_digests) != 1 or len(target_digests) != 1):
        payload["measurement_status"] = "invalidated_invariant_mismatch"
    write_json(args.output, payload)
    write_json(
        args.profiler_output,
        {
            "schema_version": 1,
            "name": "consumer-runtime-v1-profiler",
            "code_commit": _git_commit(),
            "profiles": profiler,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
