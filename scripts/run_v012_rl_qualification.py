"""Run the exact-wheel v0.12 GRPO qualification on one CUDA GPU.

The workload compiles a pinned verl v0.9-shaped profile, executes two real
Qwen3-0.6B NF4 LoRA rollout/update cycles, and records systems evidence.  Its
target-length reward is a deterministic formatting objective, not a task-
quality benchmark.
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

import yaml

from miniverl import __version__
from miniverl.algorithms.contract import UPSTREAM_VERL_COMMIT, UPSTREAM_VERL_TAG
from miniverl.bridge.rl_v09 import VERL_RL_V09_PROFILE, publish_imported_verl_rl_v09
from miniverl.config import RunConfig
from miniverl.trainer import OPDTrainer
from miniverl.trajectory.io import read_trajectories
from miniverl.utils.runs import read_jsonl, write_json_atomic

MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
PROMPTS_PER_CYCLE = 2
SAMPLES_PER_PROMPT = 4
CYCLES = 2
MAX_PEAK_RESERVED_GIB = 14.5


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_digest(trainer: OPDTrainer) -> str:
    import torch

    digest = hashlib.sha256()
    for name, tensor in sorted(trainer.student.trainable_state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.view(dtype=torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _write_dataset(path: Path) -> str:
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = []
    for index, target in enumerate((80, 100, 120, 140), start=1):
        rows.append(
            {
                "prompt": [
                    {
                        "role": "system",
                        "content": "Follow the requested response-length constraint. /no_think",
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Write one plain-text sentence about local experiment {index}. "
                            f"Aim for exactly {target} characters. /no_think"
                        ),
                    },
                ],
                "data_source": "miniverl_v012_systems_qualification",
                "ability": "target_length",
                "reward_model": {"style": "target_length", "characters": target},
                "extra_info": {"qualification_item": index, "target_characters": target},
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=PROMPTS_PER_CYCLE)
    return _sha256(path)


def _source_payload(dataset: Path) -> dict[str, Any]:
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
            "seed": 1207,
        },
        "actor_rollout_ref": {
            "model": {
                "path": MODEL_ID,
                "enable_gradient_checkpointing": True,
                "lora_rank": 16,
                "lora_alpha": 32,
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
                "optim": {"lr": 1.0e-5, "weight_decay": 0.01, "lr_warmup_steps": 0},
                "ppo_mini_batch_size": PROMPTS_PER_CYCLE * SAMPLES_PER_PROMPT,
                "ppo_epochs": 1,
                "loss_agg_mode": "token-mean",
                "clip_ratio": 0.2,
                "clip_ratio_low": 0.2,
                "clip_ratio_high": 0.2,
                "clip_ratio_c": 3.0,
                "use_kl_loss": False,
                "entropy_coeff": 0.0,
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
        "algorithm": {
            "adv_estimator": "grpo",
            "norm_adv_by_std_in_grpo": True,
            "gamma": 1.0,
            "lam": 1.0,
            "use_kl_in_reward": False,
        },
        "trainer": {
            "total_training_steps": CYCLES,
            "total_epochs": 1,
            "project_name": "mini-verl",
            "experiment_name": "v012-grpo-qualification",
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


def run(args: argparse.Namespace) -> dict[str, Any]:
    import bitsandbytes.functional as bnb_functional
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("v0.12 RL qualification requires exactly one CUDA GPU")
    if _sha256(args.wheel) != args.wheel_sha256:
        raise RuntimeError("candidate wheel checksum does not match the declared binding")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    if head != args.commit:
        raise RuntimeError("checkout HEAD does not match the candidate source commit")
    if subprocess.run(
        ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
    ).stdout:
        raise RuntimeError("v0.12 RL qualification requires a clean source checkout")

    args.work.mkdir(parents=True, exist_ok=False)
    dataset = args.work / "qualification.parquet"
    dataset_sha256 = _write_dataset(dataset)
    source = args.work / "resolved-verl-v0.9.yaml"
    source.write_text(
        yaml.safe_dump(_source_payload(dataset), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
        newline="\n",
    )
    recipe = args.work / "native.yaml"
    report = publish_imported_verl_rl_v09(
        source,
        out=recipe,
        target_verl=UPSTREAM_VERL_COMMIT,
    )
    if report.get("status") != "accepted" or report.get("executable") is not True:
        raise RuntimeError("pinned verl v0.9 profile did not compile to an executable recipe")
    config = RunConfig.from_yaml(recipe)
    trainer = OPDTrainer.from_config(
        config,
        output_dir=args.work / "runs",
        run_id="qualification",
        local_files_only=args.offline,
    )
    try:
        initial_state_sha256 = _state_digest(trainer)
        torch.cuda.reset_peak_memory_stats()
        result = trainer.train()
        final_state_sha256 = _state_digest(trainer)
        peak_reserved_gib = torch.cuda.max_memory_reserved() / 1024**3
    finally:
        trainer.close()

    if result.global_step != CYCLES or result.policy_version != CYCLES:
        raise RuntimeError("GRPO qualification did not commit two policy updates")
    if initial_state_sha256 == final_state_sha256:
        raise RuntimeError("GRPO qualification did not change the trainable actor parameters")
    if peak_reserved_gib > MAX_PEAK_RESERVED_GIB:
        raise RuntimeError("GRPO qualification exceeded the 14.5 GiB release limit")

    trajectories = read_trajectories(result.run_dir / "trajectories.jsonl")
    rewards = read_jsonl(result.run_dir / "rewards.jsonl")
    advantages = read_jsonl(result.run_dir / "advantages.jsonl")
    metrics = read_jsonl(result.run_dir / "metrics.jsonl")
    updates = [row for row in metrics if row.get("phase") == "rl"]
    cycles = [row for row in metrics if row.get("phase") == "rl_cycle"]
    expected_trajectories = CYCLES * PROMPTS_PER_CYCLE * SAMPLES_PER_PROMPT
    if not (len(trajectories) == len(rewards) == len(advantages) == expected_trajectories):
        raise RuntimeError("GRPO qualification produced an incomplete reward/trajectory set")
    if len(updates) != CYCLES or len(cycles) != CYCLES:
        raise RuntimeError("GRPO qualification produced an incomplete optimizer/cycle record")
    reward_values = [float(row["raw_reward"]) for row in rewards]
    advantage_values = [
        value for row in advantages for value in row.get("advantages", []) if math.isfinite(value)
    ]
    variances = [
        float(value)
        for cycle in cycles
        for value in (
            ((cycle.get("grouped_rollouts") or {}).get("per_group_reward_variance") or {}).get(
                "values", {}
            )
        ).values()
    ]
    if not variances or max(variances) <= 0.0:
        raise RuntimeError("GRPO qualification observed no within-prompt reward variation")
    if not advantage_values or max(map(abs, advantage_values)) <= 0.0:
        raise RuntimeError("GRPO qualification observed no nonzero policy advantage")
    if any(
        not math.isfinite(float(row["loss"]))
        or int(row["selected_positions"]) < 1
        or float((row.get("verl_rl") or {}).get("advantage_std", 0.0)) <= 0.0
        for row in updates
    ):
        raise RuntimeError("GRPO optimizer update was empty, constant, or non-finite")
    checkpoint = result.run_dir / "checkpoints" / "final"
    checkpoint_files = ("adapter.safetensors", "optimizer.safetensors", "state.json")
    if any(not (checkpoint / name).is_file() for name in checkpoint_files):
        raise RuntimeError("GRPO qualification final checkpoint is incomplete")

    bnb_functional.name2qmap.clear()
    torch._C._cuda_clearCublasWorkspaces()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    payload = {
        "schema_version": 1,
        "kind": "miniverl_v012_rl_qualification",
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
            "profile": VERL_RL_V09_PROFILE,
            "compiler_status": report["status"],
            "generated_recipe_validated": report["generated_recipe_validated"],
            "source_config_sha256": report["source_config_sha256"],
            "generated_recipe_sha256": report["generated_recipe_sha256"],
        },
        "workload": {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "algorithm": "grpo",
            "reward_provider": "builtin_target_length",
            "reward_scope": "deterministic response-format systems qualification",
            "rollout_backend": "hf_cached",
            "cycles": CYCLES,
            "prompts_per_cycle": PROMPTS_PER_CYCLE,
            "samples_per_prompt": SAMPLES_PER_PROMPT,
            "trajectories": len(trajectories),
            "generated_tokens": sum(item.generated_token_count for item in trajectories),
            "optimizer_updates": result.global_step,
            "policy_version": result.policy_version,
            "selected_positions": sum(int(row["selected_positions"]) for row in updates),
            "losses": [float(row["loss"]) for row in updates],
            "reward_min": min(reward_values),
            "reward_max": max(reward_values),
            "max_within_prompt_reward_variance": max(variances),
            "max_absolute_advantage": max(map(abs, advantage_values)),
            "initial_trainable_state_sha256": initial_state_sha256,
            "final_trainable_state_sha256": final_state_sha256,
            "checkpoint_sha256": {name: _sha256(checkpoint / name) for name in checkpoint_files},
        },
        "resource_contract": {
            "peak_reserved_gib": peak_reserved_gib,
            "limit_gib": MAX_PEAK_RESERVED_GIB,
            "peak_reserved_within_limit": True,
        },
        "inputs": {"dataset_sha256": dataset_sha256},
        "scientific_scope": {
            "runtime_correctness_only": True,
            "task_quality_evaluated": False,
            "distributed_execution_tested": False,
            "other_hardware_measured": False,
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
