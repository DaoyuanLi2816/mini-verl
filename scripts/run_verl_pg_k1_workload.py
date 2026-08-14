"""Run the bounded RTX 4080 sampled-k1 policy-gradient OPD workload.

This records systems and semantic-conformance evidence only. It does not score
task quality or compare algorithms.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from run_verl_opd_reference_workload import (
    INTERRUPT_AFTER,
    LOGICAL_BATCH,
    PROMPT_LIMIT,
    RESPONSE_LIMIT,
    UPDATES,
    _equivalence,
    _train_resumed,
    _train_uninterrupted,
    _tree_bytes,
    summarize_run,
    write_dataset,
)

from miniverl import __version__
from miniverl.bridge.opd_pg_v08 import (
    VERL_OPD_PG_K1_V08_PROFILE,
    load_verl_pg_k1_v08_source,
)
from miniverl.bridge.opd_plan import build_immutable_opd_plan, write_immutable_opd_plan
from miniverl.bridge.opd_runtime import build_system_plan
from miniverl.utils.runs import write_json_atomic

EXAMPLE = Path("examples/verl-opd-v0.8-single-gpu-pg-k1.yaml")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _driver() -> str:
    return (
        subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
        .stdout.splitlines()[0]
        .strip()
    )


def _overrides(dataset: Path) -> list[str]:
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
        "actor_rollout_ref.rollout.max_num_seqs=4",
        "trainer.experiment_name=qwen3-pg-k1-systems-workload",
        "trainer.save_freq=4",
        f"trainer.total_training_steps={UPDATES}",
        "miniverl.batching.rollout_batch_size=4",
        "miniverl.batching.teacher_score_batch_size=4",
        "miniverl.batching.update_trajectory_batch_size=1",
    ]


def build_plan(dataset: Path, path: Path) -> tuple[Any, Any]:
    compiled = load_verl_pg_k1_v08_source(
        EXAMPLE,
        overrides=_overrides(dataset),
        accept_local_reinterpretations=True,
    )
    system = build_system_plan(compiled)
    plan = build_immutable_opd_plan(compiled, source=str(EXAMPLE), system_plan=system)
    write_immutable_opd_plan(path, plan)
    from miniverl.config import RunConfig

    return plan, RunConfig.model_validate(plan.resolved_native_config)


def run_workload(out: Path, result_path: Path, *, offline: bool) -> dict[str, Any]:
    if out.exists():
        raise FileExistsError(f"output directory already exists: {out}")
    out.mkdir(parents=True)
    started = time.perf_counter()
    dataset = out / "data/reference-workload.parquet"
    dataset_identity = write_dataset(dataset)
    plan, native = build_plan(dataset, out / "plan.json")
    reference_run = _train_uninterrupted(native, plan, out / "runs", offline=offline)
    reference = summarize_run(
        reference_run["run"], construction_seconds=reference_run["construction_seconds"]
    )
    resumed_run = _train_resumed(native, plan, out / "runs", offline=offline)
    resumed = summarize_run(resumed_run["run"], construction_seconds=0.0)
    equivalence = _equivalence(reference, resumed)

    import torch

    device = torch.cuda.get_device_properties(torch.cuda.current_device())
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "single_gpu_verl_pg_k1_systems_workload",
        "status": "measured",
        "measured_at": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "miniverl_version": __version__,
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "workload_driver_sha256": _sha256(Path(__file__)),
        "profile": VERL_OPD_PG_K1_V08_PROFILE,
        "profile_identity": plan.profile_identity,
        "verl": {**plan.pinned_verl, "distributed_execution_tested": False},
        "hardware": {
            "gpu": device.name,
            "gpu_count": 1,
            "vram_gib": round(int(device.total_memory) / 1024**3, 3),
            "driver": _driver(),
            "torch": torch.__version__,
            "cuda_runtime": str(torch.version.cuda),
            "transformers": _package_version("transformers"),
            "peft": _package_version("peft"),
            "bitsandbytes": _package_version("bitsandbytes"),
        },
        "models": {
            "student": native.models.student.model_dump(mode="json"),
            "teacher": native.models.teacher.model_dump(mode="json"),
        },
        "recipe": {
            "prompts_available": dataset_identity["distinct_prompts"],
            "prompts_consumed": reference["distinct_prompts_consumed"],
            "prompt_limit": PROMPT_LIMIT,
            "response_limit": RESPONSE_LIMIT,
            "logical_batch": LOGICAL_BATCH,
            "optimizer_updates": UPDATES,
            "interrupt_after_updates": INTERRUPT_AFTER,
            "teacher_target": "sampled_token_log_probability",
            "estimator": "k1",
            "policy_loss": "vanilla",
            "compiled_plan_sha256": plan.plan_digest,
            "input_parquet_sha256": dataset_identity["sha256"],
        },
        "measurements": {
            key: value
            for key, value in reference.items()
            if key not in {"checkpoint_hashes", "checkpoint_state", "trajectory_sha256"}
        },
        "resume": {
            "resume_load_seconds": round(resumed_run["resume_load_seconds"], 4),
            "interrupt_checkpoint_bytes": resumed_run["interrupt_checkpoint_bytes"],
            **equivalence,
        },
        "artifacts": {
            "checkpoint_hashes": reference["checkpoint_hashes"],
            "trajectory_sha256": reference["trajectory_sha256"],
            "standard_peft_adapter_sha256": reference_run["adapter_manifest"]["checksums"][
                "adapter_model.safetensors"
            ],
            "total_workload_bytes": _tree_bytes(out),
        },
        "resource_contract": {
            "peak_reserved_limit_gib": 14.5,
            "peak_reserved_within_limit": reference["peak_reserved_gib"] <= 14.5,
            "total_gpu_workload_seconds": round(time.perf_counter() - started, 2),
            "gpu_hour_limit": 3.0,
        },
        "scientific_scope": {
            "runtime_correctness_only": True,
            "task_quality_evaluated": False,
            "algorithm_comparison": False,
        },
    }
    if not payload["resource_contract"]["peak_reserved_within_limit"]:
        raise RuntimeError("PG-k1 workload exceeded the 14.5 GiB reserved-memory limit")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(result_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_workload(args.out, args.result, offline=args.offline), indent=2))


if __name__ == "__main__":
    main()
