"""Qualify exact-wheel PPO, critic resume, RL export, and trained RM on one RTX 4080."""

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

import yaml

from miniverl import __version__
from miniverl.algorithms.contract import UPSTREAM_VERL_COMMIT, UPSTREAM_VERL_TAG
from miniverl.bridge.doctor import inspect_bridge_bundle
from miniverl.bridge.export import export_verl_bundle
from miniverl.bridge.rl_v09 import VERL_RL_V09_PPO_PROFILE, publish_imported_verl_rl_v09
from miniverl.config import RewardModelConfig, RunConfig
from miniverl.models.adapter_io import ADAPTER_MANIFEST, export_adapter
from miniverl.rewards import HFSequenceClassifierRewardProvider, RewardRequest
from miniverl.trainer import OPDTrainer
from miniverl.trajectory.io import read_trajectories
from miniverl.utils.runs import read_jsonl, write_json_atomic
from miniverl.utils.seeding import seed_everything

MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
RM_MODEL_ID = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
RM_REVISION = "714eb0fa89d2f80546fda750413ed43d93601a13"
PROMPTS_PER_CYCLE = 2
SAMPLES_PER_PROMPT = 2
CYCLES = 2
MAX_PEAK_RESERVED_GIB = 14.5


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _role_digest(trainer: OPDTrainer, role: str) -> str:
    import torch

    backend = trainer.student if role == "actor" else trainer.critic
    if backend is None:
        raise RuntimeError(f"missing {role} role")
    digest = hashlib.sha256()
    for name, tensor in sorted(backend.trainable_state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.view(dtype=torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _write_dataset(path: Path) -> str:
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = []
    targets = (20, 110, 24, 120)
    for index, target in enumerate(targets):
        rows.append(
            {
                "prompt": [
                    {
                        "role": "system",
                        "content": "Write one concise plain-text sentence. /no_think",
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Describe local PPO qualification item {index + 1}. "
                            f"Aim for {target} characters. /no_think"
                        ),
                    },
                ],
                "data_source": "miniverl_v013_ppo_systems_qualification",
                "ability": "target_length",
                "reward_model": {"style": "target_length", "characters": target},
                "extra_info": {"qualification_item": index + 1, "target_characters": target},
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=PROMPTS_PER_CYCLE)
    return _sha256(path)


def _source_payload(dataset: Path) -> dict[str, Any]:
    batch = PROMPTS_PER_CYCLE * SAMPLES_PER_PROMPT
    return {
        "data": {
            "train_files": [dataset.resolve().as_posix()],
            "val_files": [],
            "prompt_key": "prompt",
            "train_batch_size": PROMPTS_PER_CYCLE,
            "max_prompt_length": 128,
            "max_response_length": 32,
            "truncation": "error",
            "shuffle": False,
            "seed": 1307,
        },
        "actor_rollout_ref": {
            "model": {
                "path": MODEL_ID,
                "enable_gradient_checkpointing": True,
                "lora_rank": 8,
                "lora_alpha": 16,
                "target_modules": [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            },
            "actor": {
                "optim": {"lr": 1e-5, "weight_decay": 0.01, "lr_warmup_steps": 0},
                "ppo_mini_batch_size": batch,
                "ppo_epochs": 1,
                "loss_agg_mode": "token-mean",
                "clip_ratio": 0.2,
                "clip_ratio_low": 0.2,
                "clip_ratio_high": 0.2,
                "clip_ratio_c": 3.0,
                "use_kl_loss": False,
                "entropy_coeff": 0.001,
            },
            "rollout": {
                "name": "vllm",
                "n": SAMPLES_PER_PROMPT,
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": 0,
                "tensor_model_parallel_size": 8,
                "gpu_memory_utilization": 0.6,
            },
        },
        "critic": {
            "enable": True,
            "model": {"path": MODEL_ID},
            "optim": {"lr": 2e-5, "weight_decay": 0.01, "lr_warmup_steps": 0},
            "ppo_mini_batch_size": batch,
            "ppo_epochs": 1,
            "cliprange_value": 0.5,
        },
        "algorithm": {
            "adv_estimator": "gae",
            "norm_adv_by_std_in_grpo": True,
            "gamma": 0.99,
            "lam": 0.95,
            "use_kl_in_reward": False,
        },
        "trainer": {
            "total_training_steps": CYCLES,
            "total_epochs": 1,
            "project_name": "mini-verl",
            "experiment_name": "v013-ppo-qualification",
            "save_freq": 1,
            "test_freq": -1,
            "logger": ["console"],
            "n_gpus_per_node": 8,
            "nnodes": 2,
        },
        "miniverl": {
            "reward": {"provider": "target_length"},
            "actor": {
                "lora_enabled": True,
                "revision": MODEL_REVISION,
                "tokenizer_revision": MODEL_REVISION,
                "dtype": "auto",
                "quantization": "nf4",
            },
            "critic": {"lora_enabled": True},
            "rollout": {"backend": "hf_cached"},
            "memory": {"strategy": "resident"},
            "update": {"trajectory_batch_size": 1},
        },
    }


def _driver_version() -> str:
    return subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()[0]


def _run_training(
    config: RunConfig, root: Path, run_id: str
) -> tuple[Any, str, str, str, str, float]:
    import torch

    trainer = OPDTrainer.from_config(config, output_dir=root, run_id=run_id)
    actor_before = _role_digest(trainer, "actor")
    critic_before = _role_digest(trainer, "critic")
    try:
        torch.cuda.reset_peak_memory_stats()
        result = trainer.train()
        actor_after = _role_digest(trainer, "actor")
        critic_after = _role_digest(trainer, "critic")
        peak = torch.cuda.max_memory_reserved() / 1024**3
    finally:
        trainer.close()
    if actor_before == actor_after or critic_before == critic_after:
        raise RuntimeError("PPO qualification did not change both actor and critic")
    return result, actor_before, actor_after, critic_before, critic_after, peak


def _resume_training(config: RunConfig, root: Path) -> tuple[Any, str, str]:
    trainer = OPDTrainer.from_config(config, output_dir=root, run_id="interrupted")
    trainer.cycle = 0
    trainer._run_cycle()
    trainer.save_checkpoint(name="interrupt")
    run_root = trainer.paths.root
    trainer.close()
    resumed = OPDTrainer.from_config(config, resume=run_root)
    try:
        result = resumed.train()
        actor = _role_digest(resumed, "actor")
        critic = _role_digest(resumed, "critic")
    finally:
        resumed.close()
    return result, actor, critic


def _qualify_reward_model(*, offline: bool) -> dict[str, Any]:
    import torch

    config = RewardModelConfig(
        model_id=RM_MODEL_ID,
        revision=RM_REVISION,
        tokenizer_revision=RM_REVISION,
        batch_size=2,
        max_length=64,
        offload_between_phases=True,
    )
    provider = HFSequenceClassifierRewardProvider.load(
        config, device="cuda", local_files_only=offline
    )
    requests = [
        RewardRequest.create(
            trajectory_id=f"rm-{index}",
            prompt_group_id="rm-group",
            sample_index=index,
            samples_per_prompt=2,
            row_digest="a" * 64,
            prompt_text="Assess this response.",
            response_text=response,
            reward_model={},
            ground_truth=None,
            data_source="rm-qualification",
        )
        for index, response in enumerate(("This is excellent.", "This is terrible."))
    ]
    try:
        torch.cuda.reset_peak_memory_stats()
        provider.to_device("cuda")
        first = provider.score_batch(requests)
        second = provider.score_batch(requests)
        peak = torch.cuda.max_memory_reserved() / 1024**3
    finally:
        provider.release()
    first_scores = [float(result.raw_reward) for result in first if result.raw_reward is not None]
    second_scores = [float(result.raw_reward) for result in second if result.raw_reward is not None]
    if (
        len(first_scores) != 2
        or first_scores != second_scores
        or first_scores[0] == first_scores[1]
    ):
        raise RuntimeError("trained reward-model qualification was not deterministic/nonconstant")
    return {
        "model_id": RM_MODEL_ID,
        "revision": RM_REVISION,
        "identity": provider.identity.model_dump(mode="json"),
        "scores": first_scores,
        "deterministic_repeat": True,
        "peak_reserved_gib": peak,
        "offloaded_after_phase": provider.device == "cpu",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import bitsandbytes.functional as bnb_functional
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("PPO qualification requires exactly one CUDA GPU")
    if _sha256(args.wheel) != args.wheel_sha256:
        raise RuntimeError("candidate wheel checksum mismatch")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    if head != args.commit:
        raise RuntimeError("checkout HEAD does not match candidate source commit")
    if subprocess.run(
        ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
    ).stdout:
        raise RuntimeError("PPO qualification requires a clean source checkout")

    args.work.mkdir(parents=True, exist_ok=False)
    dataset = args.work / "qualification.parquet"
    dataset_sha256 = _write_dataset(dataset)
    source = args.work / "resolved-verl-v0.9-ppo.yaml"
    source.write_text(
        yaml.safe_dump(_source_payload(dataset), sort_keys=False), encoding="utf-8", newline="\n"
    )
    recipe = args.work / "native-ppo.yaml"
    report = publish_imported_verl_rl_v09(
        source,
        out=recipe,
        target_verl=UPSTREAM_VERL_COMMIT,
        profile=VERL_RL_V09_PPO_PROFILE,
    )
    if report.get("status") != "accepted" or report.get("executable") is not True:
        raise RuntimeError("pinned PPO profile did not compile")
    config = RunConfig.from_yaml(recipe)

    seed_everything(config.run.seed, deterministic=True)
    result, actor_before, actor_digest, critic_before, critic_digest, peak = _run_training(
        config, args.work / "runs", "reference"
    )
    seed_everything(config.run.seed, deterministic=True)
    resumed, resumed_actor, resumed_critic = _resume_training(config, args.work / "resume-runs")
    if (actor_digest, critic_digest) != (resumed_actor, resumed_critic):
        raise RuntimeError("PPO interruption/resume did not reproduce actor and critic tensors")
    if result.global_step != CYCLES or result.critic_update_count != CYCLES:
        raise RuntimeError("PPO qualification did not commit two actor and critic updates")
    if (
        resumed.global_step != result.global_step
        or resumed.critic_update_count != result.critic_update_count
    ):
        raise RuntimeError("resumed PPO counters differ from uninterrupted execution")
    if peak > MAX_PEAK_RESERVED_GIB:
        raise RuntimeError("PPO qualification exceeded the 14.5 GiB release limit")

    trajectories = read_trajectories(result.run_dir / "trajectories.jsonl")
    rewards = read_jsonl(result.run_dir / "rewards.jsonl")
    advantages = read_jsonl(result.run_dir / "advantages.jsonl")
    metrics = read_jsonl(result.run_dir / "metrics.jsonl")
    actor_updates = [row for row in metrics if row.get("phase") == "rl"]
    critic_updates = [row for row in metrics if row.get("phase") == "ppo_critic"]
    advantage_values = [
        float(value)
        for row in advantages
        for value in row.get("advantages", [])
        if math.isfinite(float(value))
    ]
    return_values = [
        float(value)
        for row in advantages
        for value in row.get("returns", [])
        if math.isfinite(float(value))
    ]
    if len(actor_updates) != CYCLES or len(critic_updates) != CYCLES:
        raise RuntimeError("PPO actor/critic metric records are incomplete")
    if not advantage_values or max(map(abs, advantage_values)) <= 0.0 or not return_values:
        raise RuntimeError("PPO qualification did not produce nontrivial advantages/returns")
    if any(not math.isfinite(float(row["loss"])) for row in actor_updates):
        raise RuntimeError("PPO actor loss is non-finite")
    if any(not math.isfinite(float(row["value_loss"])) for row in critic_updates):
        raise RuntimeError("PPO critic loss is non-finite")
    checkpoint = result.run_dir / "checkpoints" / "final"
    checkpoint_files = (
        "adapter.safetensors",
        "optimizer.safetensors",
        "critic.safetensors",
        "critic-optimizer.safetensors",
        "state.json",
        "checkpoint.json",
    )
    if any(not (checkpoint / name).is_file() for name in checkpoint_files):
        raise RuntimeError("PPO final checkpoint is incomplete")

    adapter_dir = result.run_dir / "final-peft-adapter"
    adapter_manifest, _ = export_adapter(
        result.run_dir,
        checkpoint,
        adapter_dir,
        local_files_only=args.offline,
    )
    bundle = args.work / "verl-v09-handoff"
    bundle_report = export_verl_bundle(result.run_dir, target_verl=UPSTREAM_VERL_TAG, out=bundle)
    diagnosis = inspect_bridge_bundle(bundle, require_adapter_payload=True)
    if (
        diagnosis.get("verdict") != "ok"
        or diagnosis.get("artifact_bundle_complete") is not True
        or diagnosis.get("launchable") is not False
        or (diagnosis.get("critic_checkpoint") or {}).get("status") != "ok"
    ):
        raise RuntimeError("verl v0.9 PPO handoff bundle did not pass the local doctor")

    reward_model = _qualify_reward_model(offline=args.offline)
    bnb_functional.name2qmap.clear()
    torch._C._cuda_clearCublasWorkspaces()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    payload = {
        "schema_version": 1,
        "kind": "miniverl_v013_ppo_qualification",
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
            "driver": _driver_version(),
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
        "upstream": {
            "repository": "https://github.com/verl-project/verl",
            "tag": UPSTREAM_VERL_TAG,
            "commit": UPSTREAM_VERL_COMMIT,
            "profile": VERL_RL_V09_PPO_PROFILE,
            "compiler_status": report["status"],
            "generated_recipe_sha256": report["generated_recipe_sha256"],
        },
        "workload": {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "algorithm": "ppo",
            "reward_provider": "builtin_target_length",
            "cycles": CYCLES,
            "prompts_per_cycle": PROMPTS_PER_CYCLE,
            "samples_per_prompt": SAMPLES_PER_PROMPT,
            "trajectories": len(trajectories),
            "rewards": len(rewards),
            "actor_updates": result.global_step,
            "critic_updates": result.critic_update_count,
            "max_absolute_advantage": max(map(abs, advantage_values)),
            "return_min": min(return_values),
            "return_max": max(return_values),
            "actor_losses": [float(row["loss"]) for row in actor_updates],
            "critic_losses": [float(row["value_loss"]) for row in critic_updates],
            "initial_actor_state_sha256": actor_before,
            "actor_state_sha256": actor_digest,
            "initial_critic_state_sha256": critic_before,
            "critic_state_sha256": critic_digest,
            "checkpoint_sha256": {name: _sha256(checkpoint / name) for name in checkpoint_files},
        },
        "resume": {
            "status": "exact_match",
            "actor_tensor_digest_identical": True,
            "critic_tensor_digest_identical": True,
            "actor_update_count": resumed.global_step,
            "critic_update_count": resumed.critic_update_count,
        },
        "trained_reward_model": reward_model,
        "handoff": {
            "profile": bundle_report["profile"],
            "artifact_bundle_complete": diagnosis["artifact_bundle_complete"],
            "critic_checkpoint_verified": diagnosis["critic_checkpoint"]["status"] == "ok",
            "adapter_manifest_sha256": _sha256(adapter_dir / ADAPTER_MANIFEST),
            "bundle_sha256s_sha256": _sha256(bundle / "provenance" / "SHA256SUMS"),
            "reward_implementation_complete": bundle_report["reward_implementation_complete"],
            "launchable": diagnosis["launchable"],
            "distributed_execution_tested": diagnosis["distributed_execution_tested"],
            "algorithm_semantic_parity": bundle_report["algorithm_semantic_parity"],
            "adapter_manifest": adapter_manifest,
        },
        "resource_contract": {
            "peak_reserved_gib": peak,
            "limit_gib": MAX_PEAK_RESERVED_GIB,
            "peak_reserved_within_limit": True,
        },
        "inputs": {"dataset_sha256": dataset_sha256},
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
