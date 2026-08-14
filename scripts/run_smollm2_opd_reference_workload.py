"""Run the bounded SmolLM2 direct-GKD developer workload on one CUDA GPU.

This is systems evidence only. It executes eight fresh-policy updates both
uninterrupted and with an interruption after update four, then verifies PEFT
export and a materialized pinned-verl handoff bundle. It does not score model
quality.
"""

from __future__ import annotations

import argparse
import hashlib
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
    TOP_K,
    UPDATES,
    _equivalence,
    _package_version,
    _train_resumed,
    _train_uninterrupted,
    _tree_bytes,
    summarize_run,
    write_dataset,
)

from miniverl import __version__
from miniverl.bridge.export import export_verl_bundle
from miniverl.bridge.materialize import materialize_verl_bundle
from miniverl.bridge.opd_plan import build_immutable_opd_plan, write_immutable_opd_plan
from miniverl.bridge.opd_runtime import build_system_plan
from miniverl.bridge.opd_v08 import VERL_OPD_V08_PROFILE, load_verl_opd_v08_source
from miniverl.utils.runs import write_json_atomic

EXAMPLE = Path("examples/verl-opd-v0.8-single-gpu-smollm2.yaml")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    ]


def build_plan(dataset: Path, path: Path) -> tuple[Any, Any]:
    compiled = load_verl_opd_v08_source(
        EXAMPLE,
        overrides=_overrides(dataset),
        accept_local_reinterpretations=True,
    )
    system = build_system_plan(compiled)
    plan = build_immutable_opd_plan(compiled, source=str(EXAMPLE), system_plan=system)
    write_immutable_opd_plan(path, plan)
    from miniverl.config import RunConfig

    return plan, RunConfig.model_validate(plan.resolved_native_config)


def _materialize(run: Path, out: Path, *, offline: bool) -> dict[str, Any]:
    bundle = out / "verl-bundle"
    exported = export_verl_bundle(run, target_verl="v0.8.0", out=bundle)
    if exported["launchable"] is not False:
        raise RuntimeError("an unmaterialized bundle must fail closed")
    materialized = materialize_verl_bundle(bundle, download=not offline, offline=offline)
    if materialized["launchable"] is not True:
        raise RuntimeError("the exact materialized bundle did not become launchable")
    if materialized["distributed_execution_tested"] is not False:
        raise RuntimeError("materialization cannot imply distributed execution")
    return {
        "artifact_bundle_complete": materialized["artifact_bundle_complete"],
        "upstream_config_parse_passed": materialized["upstream_config_parse_passed"],
        "model_data_load_smoke_passed": materialized["model_data_load_smoke_passed"],
        "launchable": materialized["launchable"],
        "distributed_execution_tested": materialized["distributed_execution_tested"],
        "bundle_bytes": _tree_bytes(bundle),
    }


def run_workload(out: Path, result_path: Path, *, offline: bool) -> dict[str, Any]:
    if out.exists():
        raise FileExistsError(f"output directory already exists: {out}")
    out.mkdir(parents=True)
    started = time.perf_counter()
    dataset = out / "data/smollm2-workload.parquet"
    dataset_identity = write_dataset(dataset)
    plan, native = build_plan(dataset, out / "plan.json")
    reference_run = _train_uninterrupted(native, plan, out / "runs", offline=offline)
    reference = summarize_run(
        reference_run["run"], construction_seconds=reference_run["construction_seconds"]
    )
    resumed_run = _train_resumed(native, plan, out / "runs", offline=offline)
    resumed = summarize_run(resumed_run["run"], construction_seconds=0.0)
    equivalence = _equivalence(reference, resumed)
    scaleout = _materialize(reference_run["run"], out, offline=offline)

    import torch

    device = torch.cuda.get_device_properties(torch.cuda.current_device())
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "single_gpu_smollm2_direct_gkd_developer_workload",
        "status": "maintainer_measured",
        "measured_at": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "miniverl_version": __version__,
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "workload_driver_sha256": _sha256(Path(__file__)),
        "profile": VERL_OPD_V08_PROFILE,
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
            "teacher_target": "top_k_ids_log_probs_and_mass",
            "top_k": TOP_K,
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
            "standard_peft_load_verified": True,
            "total_workload_bytes": _tree_bytes(out),
        },
        "scaleout": scaleout,
        "resource_contract": {
            "peak_reserved_limit_gib": 14.5,
            "peak_reserved_within_limit": reference["peak_reserved_gib"] <= 14.5,
            "total_gpu_workload_seconds": round(time.perf_counter() - started, 2),
            "gpu_hour_limit": 3.0,
        },
        "scientific_scope": {
            "runtime_correctness_only": True,
            "alignment_quality_evaluated": False,
            "task_quality_evaluated": False,
            "algorithm_comparison": False,
        },
    }
    if not payload["resource_contract"]["peak_reserved_within_limit"]:
        raise RuntimeError("SmolLM2 workload exceeded the 14.5 GiB reserved-memory limit")
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
