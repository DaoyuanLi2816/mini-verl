"""Run a wheel-installed, exact-commit RTX 4080 release qualification smoke."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from miniverl import __version__
from miniverl.models.adapter_io import export_adapter
from miniverl.qualification import GPUQualification, sha256_file
from miniverl.utils.runs import canonical_json, read_jsonl, write_json_atomic

PROFILE = "verl-opd-v0.8-single-gpu-v1"
BUILTIN = "builtin:qwen3-0.6b-1.7b-opd"
_PACKAGES = (
    "torch",
    "transformers",
    "peft",
    "accelerate",
    "bitsandbytes",
    "numpy",
    "pyarrow",
)


def _package_versions() -> dict[str, str]:
    return {name: importlib.metadata.version(name) for name in _PACKAGES}


def _run_cli(executable: str, cwd: Path, *arguments: str) -> dict[str, Any] | None:
    completed = subprocess.run(
        [executable, *arguments],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if "--json" not in arguments:
        return None
    output = completed.stdout.strip()
    return json.loads(output) if output else None


def _driver_version() -> str:
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return completed.stdout.splitlines()[0].strip()


def _assert_known_good(path: Path, packages: dict[str, str], gpu_name: str) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = manifest["packages"]
    differences = {
        name: {"expected": expected.get(name), "actual": actual}
        for name, actual in packages.items()
        if expected.get(name) != actual
    }
    if differences:
        raise RuntimeError(
            "installed packages do not match the known-good stack: " + canonical_json(differences)
        )
    if platform.python_version() != manifest["platform"]["python"]:
        raise RuntimeError(
            f"Python {platform.python_version()} does not match known-good "
            f"{manifest['platform']['python']}"
        )
    if gpu_name != manifest["hardware"]["gpu"]:
        raise RuntimeError(
            f"GPU {gpu_name!r} does not match measured stack {manifest['hardware']['gpu']!r}"
        )
    return manifest


def _copy_artifact(source: Path, output: Path, name: str) -> dict[str, Any]:
    target = output / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return {
        "name": name.replace("/", "_"),
        "path": name,
        "sha256": sha256_file(target),
        "bytes": target.stat().st_size,
    }


def _reload_peft_adapter(model_id: str, revision: str, adapter: Path, *, offline: bool) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        dtype=torch.float16,
        device_map={"": "cpu"},
        local_files_only=offline,
        trust_remote_code=False,
    )
    loaded = PeftModel.from_pretrained(base, adapter, is_trainable=False)
    if not loaded.peft_config:
        raise RuntimeError("standard PEFT reload produced no adapter configuration")
    del loaded, base
    gc.collect()


def _live_cuda_tensor_summary() -> list[dict[str, Any]]:
    """Return bounded diagnostics for tensors that make teardown fail closed."""
    import torch

    summary: list[dict[str, Any]] = []
    for candidate in gc.get_objects():
        try:
            if torch.is_tensor(candidate) and candidate.is_cuda:
                summary.append(
                    {
                        "shape": list(candidate.shape),
                        "dtype": str(candidate.dtype),
                        "bytes": candidate.numel() * candidate.element_size(),
                    }
                )
        except (ReferenceError, RuntimeError):
            continue
        if len(summary) >= 16:
            break
    return summary


def _clear_process_global_cuda_caches(
    torch_module: Any | None = None, bnb_functional: Any | None = None
) -> None:
    """Release third-party device caches when qualifying whole-process teardown."""
    if torch_module is None:
        import torch as torch_module
    if bnb_functional is None:
        import bitsandbytes.functional as bnb_functional

    # bitsandbytes 0.50 retains its 8-bit dynamic map on the first CUDA device
    # in a module-global dictionary. It is not trainer state, but qualification
    # models process shutdown and therefore clears this documented cache.
    bnb_functional.name2qmap.clear()
    # cuBLAS workspaces are allocator-owned rather than live tensors. Clear the
    # process-global pool so the final measurement represents user allocations.
    torch_module._C._cuda_clearCublasWorkspaces()


def run(args: argparse.Namespace) -> GPUQualification:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("GPU qualification requires torch.cuda.is_available()")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("GPU qualification requires exactly one visible CUDA GPU")
    if not args.wheel.is_file() or args.wheel.suffix != ".whl":
        raise RuntimeError(f"wheel not found: {args.wheel}")
    if args.output.exists() and any(args.output.iterdir()):
        existing = {path.resolve() for path in args.output.iterdir()}
        if args.wheel.resolve() not in existing or len(existing) != 1:
            raise RuntimeError("qualification output must be empty except for the wheel")
    args.output.mkdir(parents=True, exist_ok=True)
    args.work.mkdir(parents=True, exist_ok=False)

    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    packages = _package_versions()
    _assert_known_good(args.known_good, packages, props.name)
    known_good_sha256 = sha256_file(args.known_good)
    baseline_allocated = int(torch.cuda.memory_allocated())
    tolerance = max(2 * 1024**2, int(baseline_allocated * 0.02))

    executable = shutil.which("miniverl")
    if executable is None:
        raise RuntimeError("installed miniverl console script was not found")
    version = subprocess.run(
        [executable, "--version"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()
    if version != f"miniverl {__version__}":
        raise RuntimeError(f"unexpected installed version output: {version!r}")
    doctor = _run_cli(executable, args.work, "doctor", "--json")
    if not doctor or doctor["verdict"]["gpu_training"] is not True:
        raise RuntimeError("doctor did not qualify GPU training")

    _run_cli(
        executable,
        args.work,
        "data",
        "sample",
        "--format",
        "verl-parquet",
        "--rows",
        "4",
        "--out",
        "prompts.parquet",
    )
    overrides = [
        'data.train_files=["prompts.parquet"]',
        "data.train_batch_size=1",
        "data.max_prompt_length=128",
        "data.max_response_length=16",
        "actor_rollout_ref.actor.ppo_mini_batch_size=1",
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=192",
        "actor_rollout_ref.rollout.max_model_len=144",
        "actor_rollout_ref.rollout.max_num_batched_tokens=192",
        "actor_rollout_ref.rollout.max_num_seqs=1",
        "distillation.distillation_loss.topk=32",
        "trainer.experiment_name=exact-commit-release-smoke",
        "trainer.save_freq=1",
        "trainer.total_training_steps=1",
        "miniverl.batching.rollout_batch_size=1",
        "miniverl.batching.teacher_score_batch_size=1",
        "miniverl.batching.update_trajectory_batch_size=1",
    ]
    plan_command = [
        "plan",
        "--profile",
        PROFILE,
        "--config",
        BUILTIN,
        "--out",
        "plan.json",
        "--json",
    ]
    for override in overrides:
        plan_command.extend(("--set", override))
    _run_cli(executable, args.work, *plan_command)
    _run_cli(
        executable,
        args.work,
        "run",
        "--profile",
        PROFILE,
        "--plan",
        "plan.json",
        "--dry-run",
        "--json",
    )

    from miniverl.bridge.opd_plan import load_and_verify_immutable_opd_plan
    from miniverl.models.adapter_io import digest_tree
    from miniverl.trainer import OPDTrainer

    plan, native = load_and_verify_immutable_opd_plan(args.work / "plan.json")
    trainer = OPDTrainer.from_config(
        native,
        output_dir=args.work / "runs",
        run_id="qualification-smoke",
        local_files_only=args.offline,
    )
    with trainer:
        result = trainer.train()
        run_dir = result.run_dir
    adapter_manifest, adapter_dir = export_adapter(
        run_dir,
        run_dir / "checkpoints/final",
        args.work / "peft-adapter",
        local_files_only=args.offline,
    )
    _reload_peft_adapter(
        native.models.student.model_id,
        str(native.models.student.revision),
        adapter_dir,
        offline=args.offline,
    )
    del trainer
    _clear_process_global_cuda_caches()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    after_teardown = int(torch.cuda.memory_allocated())
    if after_teardown > baseline_allocated + tolerance:
        raise RuntimeError(
            f"CUDA teardown left {after_teardown} bytes allocated; baseline "
            f"{baseline_allocated}, tolerance {tolerance}; live tensors "
            f"{canonical_json(_live_cuda_tensor_summary())}"
        )

    metrics = read_jsonl(run_dir / "metrics.jsonl")
    cycles = [row for row in metrics if row.get("phase") == "opd_cycle"]
    updates = [row for row in metrics if row.get("phase") == "opd"]
    if not cycles or not updates or result.global_step < 1:
        raise RuntimeError("qualification did not record rollout, teacher scoring and update")
    run_manifest_path = run_dir / "manifest.json"
    run_summary = {
        "schema_version": 1,
        "status": "completed",
        "run_manifest_sha256": sha256_file(run_manifest_path),
        "execution_plan_digest": plan.plan_digest,
        "optimizer_updates": result.global_step,
        "trajectory_sha256": sha256_file(run_dir / "trajectories.jsonl"),
        "checkpoint_digest": digest_tree(run_dir / "checkpoints/final"),
    }
    run_summary_path = args.output / "run-summary.json"
    write_json_atomic(run_summary_path, run_summary)
    artifacts = [
        {
            "name": "run_summary",
            "path": run_summary_path.name,
            "sha256": sha256_file(run_summary_path),
            "bytes": run_summary_path.stat().st_size,
        },
        _copy_artifact(args.work / "prompts.parquet", args.output, "inputs/prompts.parquet"),
        _copy_artifact(
            adapter_dir / "adapter_model.safetensors",
            args.output,
            "adapter/adapter_model.safetensors",
        ),
        _copy_artifact(
            adapter_dir / "adapter_config.json", args.output, "adapter/adapter_config.json"
        ),
        _copy_artifact(
            adapter_dir / "miniverl_adapter_manifest.json",
            args.output,
            "adapter/miniverl_adapter_manifest.json",
        ),
    ]
    record = GPUQualification(
        level="release_smoke",
        status="passed",
        measured_at=datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        source_commit=args.commit,
        miniverl_version=__version__,
        wheel={"filename": args.wheel.name, "sha256": sha256_file(args.wheel)},
        profile={
            "name": plan.profile,
            "identity_digest": plan.profile_identity["digest"],
            "upstream_tag": plan.profile_identity["upstream_tag"],
            "upstream_commit": plan.profile_identity["upstream_commit"],
        },
        environment={
            "known_good_manifest_sha256": known_good_sha256,
            "gpu_name": props.name,
            "gpu_count": 1,
            "vram_gib": round(int(props.total_memory) / 1024**3, 3),
            "driver": _driver_version(),
            "cuda_runtime": str(torch.version.cuda),
            "python": platform.python_version(),
            "packages": packages,
        },
        models=[
            {
                "role": "actor",
                "model_id": plan.models["student"]["model_id"],
                "revision": plan.models["student"]["revision"],
            },
            {
                "role": "teacher",
                "model_id": plan.models["teacher"]["model_id"],
                "revision": plan.models["teacher"]["revision"],
            },
        ],
        execution={
            "rollout_completed": True,
            "teacher_scoring_completed": float(cycles[0]["teacher_scoring_seconds"]) >= 0,
            "optimizer_updates": result.global_step,
            "peft_adapter_exported": bool(adapter_manifest["checksums"]),
            "peft_adapter_reload_verified": True,
            "cuda_allocated_before_bytes": baseline_allocated,
            "cuda_allocated_after_teardown_bytes": after_teardown,
            "cuda_teardown_tolerance_bytes": tolerance,
        },
        inputs={
            "config_sha256": plan.source_config["sha256"],
            "plan_sha256": sha256_file(args.work / "plan.json"),
            "parquet_sha256": sha256_file(args.work / "prompts.parquet"),
        },
        artifacts=artifacts,
        checks={
            "executed": [
                "wheel_install",
                "doctor",
                "plan",
                "run_dry_run",
                "real_actor_teacher_update",
                "peft_reload",
                "cuda_teardown",
            ],
            "skipped": [],
            "not_applicable": ["distributed_verl_execution"],
        },
        scientific_scope={
            "runtime_correctness_only": True,
            "task_quality_evaluated": False,
            "alignment_quality_evaluated": False,
            "distributed_execution_tested": False,
            "other_hardware_measured": False,
        },
    )
    write_json_atomic(args.output / "qualification.json", record.model_dump(mode="json"))
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--known-good", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    if len(args.commit) != 40 or any(char not in "0123456789abcdef" for char in args.commit):
        parser.error("--commit must be a full lowercase 40-hex Git SHA")
    record = run(args)
    print(json.dumps(record.model_dump(mode="json"), sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
