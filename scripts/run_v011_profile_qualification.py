"""Qualify v0.11 local rollout profiles from one exact candidate wheel.

This is a bounded systems run. It executes one real Qwen3 optimizer update for
direct GKD with ``n=1``, grouped sampled-k1 with ``n=4``, and rewarded
sampled-k1 with ``n=4``. It does not evaluate task quality.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from miniverl import __version__
from miniverl.bridge.opd_pg_v08 import (
    load_verl_pg_k1_v08_source,
)
from miniverl.bridge.opd_plan import build_immutable_opd_plan, write_immutable_opd_plan
from miniverl.bridge.opd_runtime import build_system_plan
from miniverl.bridge.opd_v08 import load_verl_opd_v08_source
from miniverl.bridge.profiles import (
    VERL_OPD_PG_K1_GROUPED_V08_PROFILE,
    VERL_OPD_PG_K1_REWARDED_V08_PROFILE,
)
from miniverl.trainer import OPDTrainer
from miniverl.utils.runs import read_jsonl, write_json_atomic

DIRECT_SOURCE = "builtin:qwen3-0.6b-1.7b-opd"
PG_SOURCE = Path("examples/verl-opd-v0.8-single-gpu-pg-k1.yaml")
PROMPT_LIMIT = 128
RESPONSE_LIMIT = 64
LOGICAL_BATCH = 2
SAMPLES_PER_PROMPT = 4


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_dataset(path: Path, *, rewarded: bool) -> str:
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows: list[dict[str, Any]] = []
    for index in range(8):
        row: dict[str, Any] = {
            "prompt": [
                {"role": "system", "content": "Answer clearly and briefly."},
                {
                    "role": "user",
                    "content": f"Qualification item {index + 1}: state the integer {index + 1}.",
                },
            ],
            "data_source": "miniverl_v011_qualification",
            "ability": "short_answer",
            "extra_info": {"qualification_item": index + 1},
        }
        if rewarded:
            row["reward_model"] = {"style": "exact", "ground_truth": str(index + 1)}
        rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=2)
    return _sha256(path)


def _common_overrides(dataset: Path) -> list[str]:
    return [
        f'data.train_files=["{dataset.resolve().as_posix()}"]',
        "data.val_files=[]",
        f"data.train_batch_size={LOGICAL_BATCH}",
        f"data.max_prompt_length={PROMPT_LIMIT}",
        f"data.max_response_length={RESPONSE_LIMIT}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={LOGICAL_BATCH}",
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=768",
        "actor_rollout_ref.rollout.max_model_len=192",
        "actor_rollout_ref.rollout.max_num_batched_tokens=768",
        "actor_rollout_ref.rollout.max_num_seqs=8",
        "trainer.save_freq=1",
        "trainer.total_training_steps=1",
        "miniverl.batching.rollout_batch_size=4",
        "miniverl.batching.teacher_score_batch_size=4",
        "miniverl.batching.update_trajectory_batch_size=1",
    ]


def _build_plan(kind: str, dataset: Path, plan_path: Path) -> tuple[Any, Any]:
    overrides = _common_overrides(dataset)
    if kind == "direct_n1":
        overrides.extend(
            [
                "distillation.teacher_models.teacher_model.inference.max_model_len=192",
                "distillation.distillation_loss.topk=32",
                "trainer.experiment_name=v011-direct-n1-qualification",
            ]
        )
        compiled = load_verl_opd_v08_source(
            DIRECT_SOURCE,
            overrides=overrides,
            accept_local_reinterpretations=True,
        )
        source: str | Path = DIRECT_SOURCE
    else:
        overrides.extend(
            [
                f"actor_rollout_ref.rollout.n={SAMPLES_PER_PROMPT}",
                f"trainer.experiment_name=v011-{kind}-qualification",
            ]
        )
        rewarded = kind == "rewarded_pg_n4"
        if rewarded:
            overrides.extend(
                [
                    "distillation.distillation_loss.use_task_rewards=true",
                    "distillation.distillation_loss.task_reward_coef=1.0",
                    "distillation.distillation_loss.task_advantage_mode=group_center",
                    "distillation.distillation_loss.reward_provider=exact_answer",
                ]
            )
        compiled = load_verl_pg_k1_v08_source(
            PG_SOURCE,
            overrides=overrides,
            accept_local_reinterpretations=True,
            allow_grouped_samples=True,
            profile_name=(
                VERL_OPD_PG_K1_REWARDED_V08_PROFILE
                if rewarded
                else VERL_OPD_PG_K1_GROUPED_V08_PROFILE
            ),
            rewarded=rewarded,
        )
        source = PG_SOURCE
    system = build_system_plan(compiled)
    plan = build_immutable_opd_plan(
        compiled,
        source=source,
        system_plan=system,
        rollout_backend="hf_cached",
    )
    write_immutable_opd_plan(plan_path, plan)
    from miniverl.config import RunConfig

    return plan, RunConfig.model_validate(plan.resolved_native_config)


def _run_profile(kind: str, root: Path, *, offline: bool) -> dict[str, Any]:
    run_root = root / kind
    run_root.mkdir(parents=True, exist_ok=False)
    dataset = run_root / "data.parquet"
    dataset_sha256 = _write_dataset(dataset, rewarded=kind == "rewarded_pg_n4")
    plan, config = _build_plan(kind, dataset, run_root / "plan.json")
    trainer = OPDTrainer.from_config(
        config,
        output_dir=run_root / "runs",
        run_id="qualification",
        local_files_only=offline,
    )
    with trainer:
        write_json_atomic(
            trainer.paths.root / "local-execution-plan.json",
            plan.model_dump(mode="json"),
        )
        result = trainer.train()
    if result.global_step != 1 or result.policy_version != 1:
        raise RuntimeError(f"{kind}: expected one committed optimizer update")

    manifest = json.loads((result.run_dir / "manifest.json").read_text(encoding="utf-8"))
    runtime = manifest.get("rollout_runtime") or {}
    if runtime.get("backend") != "hf_cached":
        raise RuntimeError(f"{kind}: rollout backend was not hf_cached")
    expected_n = 1 if kind == "direct_n1" else SAMPLES_PER_PROMPT
    if runtime.get("samples_per_prompt") != expected_n:
        raise RuntimeError(f"{kind}: unexpected samples_per_prompt")

    trajectories = read_jsonl(result.run_dir / "trajectories.jsonl")
    expected_trajectories = LOGICAL_BATCH * expected_n
    if len(trajectories) != expected_trajectories:
        raise RuntimeError(f"{kind}: incomplete trajectory group")
    if any(row.get("samples_per_prompt") != expected_n for row in trajectories):
        raise RuntimeError(f"{kind}: trajectory group identity mismatch")
    if (
        kind != "direct_n1"
        and len({(row["prompt_group_id"], row["sample_index"]) for row in trajectories})
        != expected_trajectories
    ):
        raise RuntimeError(f"{kind}: duplicate grouped sample identity")
    if kind != "direct_n1" and any(
        len((row.get("metadata") or {}).get("actor_rollout_log_probs") or [])
        != (row.get("metadata") or {}).get("response_token_count")
        for row in trajectories
    ):
        raise RuntimeError(f"{kind}: sampled actor log-probabilities are incomplete")

    metrics = read_jsonl(result.run_dir / "metrics.jsonl")
    update = next(row for row in metrics if row.get("phase") == "opd")
    cycle = next(row for row in metrics if row.get("phase") == "opd_cycle")
    if not math.isfinite(float(update["loss"])) or int(update["selected_positions"]) < 1:
        raise RuntimeError(f"{kind}: optimizer update is not finite and non-empty")
    grouped = cycle.get("grouped_rollouts") or {}
    if grouped.get("samples_per_prompt") != expected_n:
        raise RuntimeError(f"{kind}: grouped metrics do not match the profile")

    reward_summary: dict[str, Any] = {"status": "not_applicable"}
    if kind == "rewarded_pg_n4":
        rewards = read_jsonl(result.run_dir / "rewards.jsonl")
        advantages = read_jsonl(result.run_dir / "advantages.jsonl")
        if len(rewards) != expected_trajectories or len(advantages) != expected_trajectories:
            raise RuntimeError("rewarded_pg_n4: reward/advantage records are incomplete")
        if any(row.get("status") != "ok" for row in rewards):
            raise RuntimeError("rewarded_pg_n4: reward provider did not complete")
        if any(not math.isfinite(float(row["task_advantage"])) for row in advantages):
            raise RuntimeError("rewarded_pg_n4: task advantage is not finite")
        reward_summary = {
            "status": "completed",
            "records": len(rewards),
            "provider_versions": sorted({row["provider"]["version"] for row in rewards}),
            "advantage_versions": sorted({row["implementation_version"] for row in advantages}),
            "nonzero_raw_rewards": sum(float(row["raw_reward"]) != 0.0 for row in rewards),
        }

    checkpoint = result.run_dir / "checkpoints" / "final"
    return {
        "profile": plan.profile,
        "profile_identity": plan.profile_identity,
        "plan_sha256": plan.plan_digest,
        "dataset_sha256": dataset_sha256,
        "rollout_backend": runtime["backend"],
        "backend_version": runtime["capabilities"]["backend_version"],
        "samples_per_prompt": expected_n,
        "prompt_groups": grouped.get("groups"),
        "trajectories": len(trajectories),
        "generated_tokens": grouped.get("generated_tokens"),
        "optimizer_updates": result.global_step,
        "policy_version": result.policy_version,
        "selected_positions": int(update["selected_positions"]),
        "loss_finite": True,
        "peak_reserved_gib": max(
            float((row.get("memory") or {}).get("peak_reserved_gib") or 0.0) for row in metrics
        ),
        "checkpoint_sha256": {
            name: _sha256(checkpoint / name)
            for name in ("adapter.safetensors", "optimizer.safetensors", "state.json")
        },
        "reward": reward_summary,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import bitsandbytes.functional as bnb_functional
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("v0.11 profile qualification requires exactly one CUDA GPU")
    if _sha256(args.wheel) != args.wheel_sha256:
        raise RuntimeError("candidate wheel checksum does not match the declared binding")
    args.work.mkdir(parents=True, exist_ok=False)
    baseline_allocated = int(torch.cuda.memory_allocated())
    profiles = {
        kind: _run_profile(kind, args.work, offline=args.offline)
        for kind in ("direct_n1", "grouped_pg_n4", "rewarded_pg_n4")
    }
    bnb_functional.name2qmap.clear()
    torch._C._cuda_clearCublasWorkspaces()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    residual_allocated = int(torch.cuda.memory_allocated())
    tolerance = max(2 * 1024**2, int(baseline_allocated * 0.02))
    if residual_allocated > baseline_allocated + tolerance:
        raise RuntimeError("v0.11 profile qualification left live CUDA allocations")
    payload = {
        "schema_version": 1,
        "kind": "miniverl_v011_profile_qualification",
        "status": "passed",
        "measured_at": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "source_commit": args.commit,
        "miniverl_version": __version__,
        "wheel_sha256": args.wheel_sha256,
        "hardware": {
            "gpu": torch.cuda.get_device_name(0),
            "gpu_count": 1,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cuda_runtime": str(torch.version.cuda),
            "driver": subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.splitlines()[0],
            "packages": {
                name: importlib.metadata.version(name)
                for name in (
                    "torch",
                    "transformers",
                    "peft",
                    "accelerate",
                    "bitsandbytes",
                    "numpy",
                    "pyarrow",
                    "safetensors",
                )
            },
        },
        "profiles": profiles,
        "cuda_teardown": {
            "allocated_before_bytes": baseline_allocated,
            "allocated_after_bytes": residual_allocated,
            "tolerance_bytes": tolerance,
            "passed": True,
        },
        "scientific_scope": {
            "runtime_correctness_only": True,
            "task_quality_evaluated": False,
            "distributed_execution_tested": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--wheel-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args), sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
