"""Run the bounded RTX 4080 verl-style OPD developer workload.

This is systems evidence, not an alignment or task-quality benchmark.  It
executes the exact Qwen3 profile twice: one uninterrupted eight-update run and
one four-update interruption followed by checkpoint resume.  The published
record contains only portable identities and aggregate runtime measurements.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from miniverl import __version__
from miniverl.bridge.opd_plan import build_immutable_opd_plan, write_immutable_opd_plan
from miniverl.bridge.opd_runtime import build_system_plan
from miniverl.bridge.opd_v08 import load_verl_opd_v08_source
from miniverl.models.adapter_io import export_adapter
from miniverl.trainer import OPDTrainer
from miniverl.utils.runs import read_jsonl, write_json_atomic

PROFILE = "verl-opd-v0.8-single-gpu-v1"
BUILTIN = "builtin:qwen3-0.6b-1.7b-opd"
PROMPT_ROWS = 64
UPDATES = 8
INTERRUPT_AFTER = 4
LOGICAL_BATCH = 4
PROMPT_LIMIT = 128
RESPONSE_LIMIT = 64
TOP_K = 32


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _driver_sha256() -> str:
    return _sha256(Path(__file__).resolve())


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _driver_version() -> str:
    try:
        return (
            subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            .stdout.splitlines()[0]
            .strip()
        )
    except (FileNotFoundError, IndexError, OSError, subprocess.SubprocessError):
        return "unavailable"


def _sample_records() -> list[dict[str, Any]]:
    prompts = [
        "Explain why immutable revisions help reproduce a model run.",
        "State one reason to inspect a compatibility report before execution.",
        "Describe one benefit of a checksummed training artifact.",
        "Name one safe response to a CUDA out-of-memory error.",
        "Explain token-mean loss aggregation in one sentence.",
        "State one limitation of a single-GPU runtime.",
        "Explain why a teacher target cache needs policy-version provenance.",
        "Describe the difference between a logical and physical batch.",
    ]
    rows: list[dict[str, Any]] = []
    for index in range(PROMPT_ROWS):
        family = prompts[index % len(prompts)]
        rows.append(
            {
                "prompt": [
                    {"role": "system", "content": "Answer clearly and briefly."},
                    {"role": "user", "content": f"Workload item {index + 1:02d}: {family}"},
                ],
                "data_source": "miniverl_systems_workload",
                "ability": "short_answer",
                "extra_info": {"workload_item": index + 1},
            }
        )
    return rows


def write_dataset(path: Path) -> dict[str, Any]:
    """Write the deterministic, reward-free Parquet workload."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    records = _sample_records()
    prompts = {json.dumps(row["prompt"], sort_keys=True) for row in records}
    if len(prompts) != PROMPT_ROWS:  # pragma: no cover - constant-data guard
        raise RuntimeError("reference workload prompts are not distinct")
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(records), path)
    return {"rows": len(records), "distinct_prompts": len(prompts), "sha256": _sha256(path)}


def _overrides(dataset: Path) -> list[str]:
    data_path = dataset.resolve().as_posix()
    return [
        f'data.train_files=["{data_path}"]',
        f"data.train_batch_size={LOGICAL_BATCH}",
        f"data.max_prompt_length={PROMPT_LIMIT}",
        f"data.max_response_length={RESPONSE_LIMIT}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={LOGICAL_BATCH}",
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=768",
        "actor_rollout_ref.rollout.max_model_len=192",
        "actor_rollout_ref.rollout.max_num_batched_tokens=768",
        "actor_rollout_ref.rollout.max_num_seqs=4",
        f"distillation.distillation_loss.topk={TOP_K}",
        "trainer.experiment_name=qwen3-opd-developer-workload",
        "trainer.save_freq=4",
        f"trainer.total_training_steps={UPDATES}",
        "miniverl.batching.rollout_batch_size=4",
        "miniverl.batching.teacher_score_batch_size=4",
        "miniverl.batching.update_trajectory_batch_size=1",
    ]


def build_plan(dataset: Path, path: Path) -> tuple[Any, Any]:
    compiled = load_verl_opd_v08_source(
        BUILTIN,
        overrides=_overrides(dataset),
        accept_local_reinterpretations=True,
    )
    system = build_system_plan(compiled)
    plan = build_immutable_opd_plan(compiled, source=BUILTIN, system_plan=system)
    write_immutable_opd_plan(path, plan)
    from miniverl.config import RunConfig

    native = RunConfig.model_validate(plan.resolved_native_config)
    return plan, native


def _write_plan_artifacts(trainer: OPDTrainer, plan: Any) -> None:
    write_json_atomic(
        trainer.paths.root / "local-execution-plan.json", plan.model_dump(mode="json")
    )
    write_json_atomic(
        trainer.paths.root / "verl-source-config.json",
        plan.compiled_plan["source"],
    )
    write_json_atomic(
        trainer.paths.root / "verl-compatibility-report.json",
        plan.compiled_plan,
    )


def _train_uninterrupted(native: Any, plan: Any, runs: Path, *, offline: bool) -> dict[str, Any]:
    constructed = time.perf_counter()
    trainer = OPDTrainer.from_config(
        native,
        output_dir=runs,
        run_id="uninterrupted",
        local_files_only=offline,
    )
    construction_seconds = time.perf_counter() - constructed
    with trainer:
        _write_plan_artifacts(trainer, plan)
        result = trainer.train()
    adapter_manifest, _ = export_adapter(
        result.run_dir,
        result.run_dir / "checkpoints" / "final",
        result.run_dir / "final-peft-adapter",
        local_files_only=offline,
    )
    return {
        "run": result.run_dir,
        "construction_seconds": construction_seconds,
        "result": result,
        "adapter_manifest": adapter_manifest,
    }


def _train_resumed(native: Any, plan: Any, runs: Path, *, offline: bool) -> dict[str, Any]:
    first = OPDTrainer.from_config(
        native,
        output_dir=runs,
        run_id="resumed",
        local_files_only=offline,
    )
    with first:
        _write_plan_artifacts(first, plan)
        for cycle in range(INTERRUPT_AFTER):
            first.cycle = cycle
            first._run_cycle()
        interrupt_checkpoint = first.save_checkpoint(name="interrupt")
    resume_started = time.perf_counter()
    second = OPDTrainer.from_config(native, resume=runs / "resumed", local_files_only=offline)
    resume_load_seconds = time.perf_counter() - resume_started
    with second:
        result = second.train()
    adapter_manifest, _ = export_adapter(
        result.run_dir,
        result.run_dir / "checkpoints" / "final",
        result.run_dir / "final-peft-adapter",
        local_files_only=offline,
    )
    return {
        "run": result.run_dir,
        "result": result,
        "resume_load_seconds": resume_load_seconds,
        "interrupt_checkpoint_bytes": _tree_bytes(interrupt_checkpoint),
        "adapter_manifest": adapter_manifest,
    }


def _median(values: list[float]) -> float:
    return round(float(statistics.median(values)), 4)


def summarize_run(run: Path, *, construction_seconds: float) -> dict[str, Any]:
    """Aggregate phase measurements without interpreting task quality."""
    rows = read_jsonl(run / "metrics.jsonl")
    cycles = [row for row in rows if row.get("phase") == "opd_cycle"]
    updates = [row for row in rows if row.get("phase") == "opd"]
    if len(cycles) != UPDATES or len(updates) != UPDATES:
        raise RuntimeError(
            f"expected {UPDATES} cycle/update rows, got {len(cycles)}/{len(updates)}"
        )
    steady_cycles = cycles[1:]
    steady_updates = updates[1:]
    first = cycles[0]
    generated = [float(row["rollouts"]["generated_tokens"]) for row in steady_cycles]
    selected = [float(row["selected_positions"]) for row in steady_updates]
    all_memory = [row.get("memory") or {} for row in rows]
    rollout_execution = [row.get("rollout_execution") or {} for row in cycles]
    trajectories = read_jsonl(run / "trajectories.jsonl")
    distinct = {str((row.get("metadata") or {}).get("row_digest")) for row in trajectories}
    checkpoint = run / "checkpoints" / "final"
    adapter = run / "final-peft-adapter"
    cache = run / "teacher-cache"
    return {
        "cold_startup_seconds": round(construction_seconds, 4),
        "time_to_first_rollout_seconds": round(
            construction_seconds + float(first["rollout_seconds"]), 4
        ),
        "time_to_first_teacher_targets_seconds": round(
            construction_seconds
            + float(first["rollout_seconds"])
            + float(first["teacher_scoring_seconds"]),
            4,
        ),
        "time_to_first_update_seconds": round(construction_seconds + float(first["seconds"]), 4),
        "steady_state_median_seconds": {
            "rollout": _median([float(row["rollout_seconds"]) for row in steady_cycles]),
            "teacher_scoring": _median(
                [float(row["teacher_scoring_seconds"]) for row in steady_cycles]
            ),
            "actor_update": _median([float(row["seconds"]) for row in steady_updates]),
        },
        "steady_state_median_throughput": {
            "rollout_tokens_per_second": _median(
                [
                    tokens / float(row["rollout_seconds"])
                    for tokens, row in zip(generated, steady_cycles, strict=True)
                ]
            ),
            "teacher_scored_positions_per_second": _median(
                [float(row["teacher_scored_positions_per_second"]) for row in steady_cycles]
            ),
            "update_positions_per_second": _median(
                [
                    positions / float(row["seconds"])
                    for positions, row in zip(selected, steady_updates, strict=True)
                ]
            ),
        },
        "peak_allocated_gib": round(
            max(float(item.get("peak_allocated_gib") or 0.0) for item in all_memory), 4
        ),
        "peak_reserved_gib": round(
            max(float(item.get("peak_reserved_gib") or 0.0) for item in all_memory), 4
        ),
        "batch_downshifts": {
            "rollout_oom": sum(int(item.get("oom_downshifts") or 0) for item in rollout_execution),
            "update_chunk_oom": sum(
                1
                for row in read_jsonl(run / "events.jsonl")
                if row.get("event") == "oom_chunk_retry"
            ),
        },
        "observed_rollout_physical_batch_sizes": sorted(
            {
                int(size)
                for item in rollout_execution
                for size in item.get("physical_batch_sizes", [])
            }
        ),
        "prompts_consumed": len(trajectories),
        "distinct_prompts_consumed": len(distinct),
        "cache_bytes": _tree_bytes(cache),
        "checkpoint_bytes": _tree_bytes(checkpoint),
        "adapter_bytes": _tree_bytes(adapter),
        "total_run_bytes": _tree_bytes(run),
        "checkpoint_hashes": {
            name: _sha256(checkpoint / name)
            for name in ("adapter.safetensors", "optimizer.safetensors", "state.json")
        },
        "checkpoint_state": json.loads((checkpoint / "state.json").read_text(encoding="utf-8")),
        "trajectory_sha256": _sha256(run / "trajectories.jsonl"),
    }


def _equivalence(reference: dict[str, Any], resumed: dict[str, Any]) -> dict[str, Any]:
    reference_hashes = reference["checkpoint_hashes"]
    resumed_hashes = resumed["checkpoint_hashes"]
    tensor_match = all(
        reference_hashes[name] == resumed_hashes[name]
        for name in ("adapter.safetensors", "optimizer.safetensors")
    )
    reference_state = reference["checkpoint_state"]
    resumed_state = resumed["checkpoint_state"]
    # The resolved config includes the run id, so that identity digest must
    # differ between separately named reference and resumed runs. Every actual
    # training-state field still has to match exactly.
    state_fields_match = all(
        reference_state[key] == resumed_state[key]
        for key in reference_state
        if key != "resolved_config_digest"
    )
    trajectory_match = reference["trajectory_sha256"] == resumed["trajectory_sha256"]
    if not tensor_match or not state_fields_match or not trajectory_match:
        raise RuntimeError("uninterrupted and resumed executions did not match exactly")
    return {
        "status": "exact_match",
        "adapter_and_optimizer_byte_identical": tensor_match,
        "training_state_fields_identical": state_fields_match,
        "excluded_run_identity_field": "resolved_config_digest",
        "trajectories_byte_identical": trajectory_match,
        "global_optimizer_steps": UPDATES,
        "task_cursor": UPDATES * LOGICAL_BATCH,
    }


def run_workload(out: Path, result_path: Path, *, offline: bool) -> dict[str, Any]:
    if out.exists():
        raise FileExistsError(f"output directory already exists: {out}")
    out.mkdir(parents=True)
    started = time.perf_counter()
    dataset = out / "data" / "reference-workload.parquet"
    dataset_identity = write_dataset(dataset)
    plan_path = out / "plan.json"
    plan, native = build_plan(dataset, plan_path)
    runs = out / "runs"
    uninterrupted = _train_uninterrupted(native, plan, runs, offline=offline)
    reference = summarize_run(
        uninterrupted["run"], construction_seconds=uninterrupted["construction_seconds"]
    )
    resumed_run = _train_resumed(native, plan, runs, offline=offline)
    resumed = summarize_run(resumed_run["run"], construction_seconds=0.0)
    equivalence = _equivalence(reference, resumed)

    import torch

    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    artifact = {
        "schema_version": 1,
        "kind": "single_gpu_opd_developer_workload",
        "status": "measured",
        "measured_at": (
            datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        ),
        "miniverl_version": __version__,
        "source_commit": _git_head(),
        "workload_driver_sha256": _driver_sha256(),
        "profile": PROFILE,
        "verl": {
            "tag": plan.pinned_verl["tag"],
            "commit": plan.pinned_verl["commit"],
            "distributed_execution_tested": False,
        },
        "hardware": {
            "gpu": properties.name,
            "gpu_count": 1,
            "vram_gib": round(int(properties.total_memory) / (1024**3), 3),
            "driver": _driver_version(),
            "torch": torch.__version__,
            "cuda_runtime": str(torch.version.cuda),
            "transformers": _package_version("transformers"),
            "peft": _package_version("peft"),
            "bitsandbytes": _package_version("bitsandbytes"),
        },
        "models": {
            "student": {
                "id": native.models.student.model_id,
                "revision": native.models.student.revision,
                "quantization": native.models.student.quantization.value,
                "adapter": "lora-r8-alpha16",
            },
            "teacher": {
                "id": native.models.teacher.model_id,
                "revision": native.models.teacher.revision,
                "quantization": native.models.teacher.quantization.value,
            },
        },
        "recipe": {
            "runtime_strategy": "dual_model_resident",
            "dataset_rows": dataset_identity["rows"],
            "distinct_dataset_prompts": dataset_identity["distinct_prompts"],
            "prompts_consumed": reference["prompts_consumed"],
            "distinct_prompts_consumed": reference["distinct_prompts_consumed"],
            "prompt_limit": PROMPT_LIMIT,
            "response_limit": RESPONSE_LIMIT,
            "logical_batch": LOGICAL_BATCH,
            "rollout_physical_batch": 4,
            "teacher_score_batch": 4,
            "update_physical_batch": 1,
            "top_k": TOP_K,
            "optimizer_updates": UPDATES,
            "compiled_plan_sha256": plan.plan_digest,
            "input_parquet_sha256": dataset_identity["sha256"],
        },
        "measurements": {
            key: value
            for key, value in reference.items()
            if key not in {"checkpoint_hashes", "checkpoint_state", "trajectory_sha256"}
        },
        "resume": {
            "interrupt_after_optimizer_updates": INTERRUPT_AFTER,
            "resume_load_seconds": round(resumed_run["resume_load_seconds"], 4),
            "interrupt_checkpoint_bytes": resumed_run["interrupt_checkpoint_bytes"],
            **equivalence,
        },
        "artifacts": {
            "checkpoint_hashes": reference["checkpoint_hashes"],
            "trajectory_sha256": reference["trajectory_sha256"],
            "standard_peft_adapter_sha256": uninterrupted["adapter_manifest"]["checksums"][
                "adapter_model.safetensors"
            ],
            "standard_peft_load_verified": True,
            "workload_output_bytes": _tree_bytes(out),
        },
        "resource_contract": {
            "peak_reserved_limit_gib": 14.5,
            "peak_reserved_within_limit": reference["peak_reserved_gib"] <= 14.5,
            "total_gpu_workload_seconds": round(time.perf_counter() - started, 2),
            "gpu_hour_limit": 4.0,
        },
        "scientific_scope": {
            "runtime_correctness_only": True,
            "alignment_quality_evaluated": False,
            "task_quality_evaluated": False,
            "opd_beats_sft_dpo_or_kd_claimed": False,
        },
    }
    if not artifact["resource_contract"]["peak_reserved_within_limit"]:
        raise RuntimeError("reference workload exceeded the 14.5 GiB reserved-memory limit")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(result_path, artifact)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    payload = run_workload(args.out.resolve(), args.result.resolve(), offline=args.offline)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
