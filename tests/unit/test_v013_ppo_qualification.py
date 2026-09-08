from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml


def test_v013_gpu_workload_compiles_to_executable_ppo(tmp_path: Path) -> None:
    from miniverl.algorithms.contract import UPSTREAM_VERL_COMMIT
    from miniverl.bridge.rl_v09 import VERL_RL_V09_PPO_PROFILE, publish_imported_verl_rl_v09
    from miniverl.config import RunConfig
    from miniverl.data.verl_parquet import VerlParquetDataset
    from scripts.run_v013_ppo_qualification import _source_payload, _write_dataset

    dataset = tmp_path / "data.parquet"
    _write_dataset(dataset)
    source = tmp_path / "source.yaml"
    source.write_text(yaml.safe_dump(_source_payload(dataset), sort_keys=False), encoding="utf-8")
    recipe = tmp_path / "native.yaml"

    report = publish_imported_verl_rl_v09(
        source,
        out=recipe,
        target_verl=UPSTREAM_VERL_COMMIT,
        profile=VERL_RL_V09_PPO_PROFILE,
    )
    config = RunConfig.from_yaml(recipe)

    assert report["status"] == "accepted"
    assert config.algorithm.name.value == "ppo"
    assert config.critic.enabled is True
    assert config.algorithm.entropy_coeff > 0.0
    assert config.reward.provider.value == "target_length"
    assert config.rollout.samples_per_prompt == 2
    assert config.train.cycles == 2
    assert VerlParquetDataset(config.source).inspect().rows == {"train": 4, "val": 0}


def _qualification_payload() -> tuple[dict, SimpleNamespace]:
    digest = "a" * 64
    environment = SimpleNamespace(
        gpu_name="NVIDIA GeForce RTX 4080",
        python="3.12.13",
        cuda_runtime="12.8",
        driver="590.48.01",
        packages={"torch": "2.10.0"},
    )
    qualification = SimpleNamespace(
        source_commit="b" * 40,
        miniverl_version="0.13.0",
        wheel=SimpleNamespace(sha256="c" * 64),
        environment=environment,
    )
    checkpoint = dict.fromkeys(
        (
            "adapter.safetensors",
            "optimizer.safetensors",
            "critic.safetensors",
            "critic-optimizer.safetensors",
            "state.json",
            "checkpoint.json",
        ),
        digest,
    )
    payload = {
        "schema_version": 1,
        "kind": "miniverl_v013_ppo_qualification",
        "status": "passed",
        "source_commit": qualification.source_commit,
        "miniverl_version": qualification.miniverl_version,
        "wheel_sha256": qualification.wheel.sha256,
        "hardware": {
            "gpu": environment.gpu_name,
            "gpu_count": 1,
            "platform": "Linux-Microsoft-WSL2",
            "python": environment.python,
            "cuda_runtime": environment.cuda_runtime,
            "driver": environment.driver,
            "packages": environment.packages,
        },
        "upstream": {
            "tag": "v0.9.0",
            "commit": "483b8a009ba3a97563edee3a19887e4862b8094a",
            "profile": "verl-rl-v0.9-single-gpu-v2",
            "compiler_status": "accepted",
            "generated_recipe_sha256": digest,
        },
        "workload": {
            "model_id": "Qwen/Qwen3-0.6B",
            "model_revision": "c1899de289a04d12100db370d81485cdf75e47ca",
            "algorithm": "ppo",
            "reward_provider": "builtin_target_length",
            "cycles": 2,
            "prompts_per_cycle": 2,
            "samples_per_prompt": 2,
            "trajectories": 8,
            "rewards": 8,
            "actor_updates": 2,
            "critic_updates": 2,
            "max_absolute_advantage": 0.5,
            "return_min": -0.2,
            "return_max": 0.8,
            "actor_losses": [0.1, 0.2],
            "critic_losses": [0.3, 0.4],
            "initial_actor_state_sha256": "1" * 64,
            "actor_state_sha256": "2" * 64,
            "initial_critic_state_sha256": "3" * 64,
            "critic_state_sha256": "4" * 64,
            "checkpoint_sha256": checkpoint,
        },
        "resume": {
            "status": "exact_match",
            "actor_tensor_digest_identical": True,
            "critic_tensor_digest_identical": True,
            "actor_update_count": 2,
            "critic_update_count": 2,
        },
        "trained_reward_model": {
            "model_id": "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
            "revision": "714eb0fa89d2f80546fda750413ed43d93601a13",
            "identity": {"name": "hf_sequence_classifier"},
            "scores": [0.9, 0.1],
            "deterministic_repeat": True,
            "peak_reserved_gib": 0.2,
            "offloaded_after_phase": True,
        },
        "handoff": {
            "profile": "verl-rl-v0.9-single-gpu-v2",
            "artifact_bundle_complete": True,
            "critic_checkpoint_verified": True,
            "adapter_manifest_sha256": digest,
            "bundle_sha256s_sha256": digest,
            "reward_implementation_complete": False,
            "launchable": False,
            "distributed_execution_tested": False,
            "algorithm_semantic_parity": True,
        },
        "resource_contract": {
            "peak_reserved_gib": 12.0,
            "peak_reserved_within_limit": True,
        },
        "scientific_scope": {
            "runtime_correctness_only": True,
            "task_quality_evaluated": False,
            "distributed_execution_tested": False,
        },
    }
    return payload, qualification


def test_v013_evidence_validator_requires_both_role_updates() -> None:
    from scripts.promote_full_gpu_qualification import _validate_v013_ppo

    payload, qualification = _qualification_payload()
    _validate_v013_ppo(payload, qualification)  # type: ignore[arg-type]

    payload["workload"]["critic_state_sha256"] = payload["workload"]["initial_critic_state_sha256"]
    with pytest.raises(ValueError, match="critic parameters did not change"):
        _validate_v013_ppo(payload, qualification)  # type: ignore[arg-type]


def test_v013_evidence_validator_rejects_inexact_resume() -> None:
    from scripts.promote_full_gpu_qualification import _validate_v013_ppo

    payload, qualification = _qualification_payload()
    payload["resume"]["critic_tensor_digest_identical"] = False
    with pytest.raises(ValueError, match="exact actor/critic resume"):
        _validate_v013_ppo(payload, qualification)  # type: ignore[arg-type]
