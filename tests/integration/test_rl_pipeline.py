# ruff: noqa: E402 - optional torch dependency is checked before torch-backed imports
"""End-to-end teacher-free grouped RL over a verl-style Parquet prompt."""

from __future__ import annotations

import json
from typing import Any

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
pytest.importorskip("torch")

from miniverl.algorithms.contract import ADVANTAGE_IMPLEMENTATION_VERSION
from miniverl.rewards import RewardProviderIdentity, RewardResult, RewardStatus

pytestmark = pytest.mark.torch


class _SampleIndexReward:
    identity = RewardProviderIdentity(
        name="test_sample_index",
        version="v1",
        config_digest="1" * 64,
        deterministic=True,
    )

    def score(self, request):  # type: ignore[no-untyped-def]
        return RewardResult(
            trajectory_id=request.trajectory_id,
            prompt_group_id=request.prompt_group_id,
            sample_index=request.sample_index,
            samples_per_prompt=request.samples_per_prompt,
            provider=self.identity,
            input_digest=request.input_digest,
            raw_reward=float(request.sample_index),
            status=RewardStatus.OK,
            duration_ms=0.0,
            deterministic=True,
        )


def _trainable_state(trainer) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in trainer.student.trainable_state_dict().items()
    }


def _critic_state(trainer) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    assert trainer.critic is not None
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in trainer.critic.trainable_state_dict().items()
    }


@pytest.mark.parametrize("algorithm", ["grpo", "dr_grpo", "rloo", "reinforce_plus_plus"])
def test_teacher_free_rl_executes_current_policy_rollout_reward_and_update(
    tmp_path, algorithm: str
) -> None:  # type: ignore[no-untyped-def]
    from miniverl.config import RunConfig
    from miniverl.trainer import OPDTrainer

    train = tmp_path / "train.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "prompt": "Choose one token.",
                    "data_source": "unit",
                    "reward_model": {"style": "exact", "ground_truth": "unused"},
                    "extra_info": {"ground_truth": "unused"},
                }
            ]
        ),
        train,
    )
    config = RunConfig.model_validate(
        {
            "run": {
                "name": "teacher-free-grpo",
                "mode": "rl",
                "seed": 17,
                "output_dir": str(tmp_path / "runs"),
            },
            "models": {
                "backend": "toy",
                "device": "cpu",
                "student": {
                    "model_id": "toy-actor",
                    "lora": {"enabled": False},
                    "toy": {
                        "hidden_size": 16,
                        "num_layers": 1,
                        "num_heads": 2,
                        "intermediate_size": 32,
                        "max_position_embeddings": 128,
                    },
                },
            },
            "source": {
                "kind": "verl_parquet",
                "train_files": [str(train)],
                "allow_plain_string_prompts": True,
                "shuffle": False,
                "use_task_rewards": True,
            },
            "rollout": {
                "backend": "hf_cached",
                "samples_per_prompt": 2,
                "temperature": 1.0,
                "max_turns": 1,
                "max_new_tokens_per_turn": 3,
                "max_total_tokens": 64,
                "record_logprobs": True,
            },
            "selection": {"selector": "all_model_tokens"},
            "loss": {
                "mode": "verl_rl_policy",
                "aggregation": "token-mean",
                "scale_by_temperature_squared": False,
                "chunk_size": 16,
            },
            "algorithm": {
                "name": algorithm,
                "implementation_version": ADVANTAGE_IMPLEMENTATION_VERSION,
            },
            "reward": {"enabled": True, "provider": "python_api"},
            "train": {
                "cycles": 1,
                "rollouts_per_cycle": 1,
                "gradient_accumulation_steps": 2,
                "learning_rate": 0.001,
            },
            "memory": {"strategy": "resident"},
            "eval": {"enabled": False},
            "report": {"enabled": False},
        }
    )

    trainer = OPDTrainer.from_config(
        config,
        run_id="grpo-e2e",
        reward_provider=_SampleIndexReward(),
    )
    try:
        result = trainer.train()
    finally:
        trainer.close()

    assert result.mode == "rl"
    assert result.global_step == 1
    rows = [
        json.loads(line) for line in (result.run_dir / "metrics.jsonl").read_text().splitlines()
    ]
    update = next(row for row in rows if row.get("phase") == "rl")
    assert update["verl_rl"]["algorithm"] == algorithm
    assert update["verl_rl"]["ratio_mean"] == pytest.approx(1.0)
    assert update["verl_rl"]["actor_entropy"] > 0.0
    advantages = [
        json.loads(line) for line in (result.run_dir / "advantages.jsonl").read_text().splitlines()
    ]
    assert len(advantages) == 2
    assert {row["name"] for row in advantages} == {algorithm}


def test_grouped_rl_preserves_agent_tool_context_and_uses_recorded_verifier(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from miniverl.config import RunConfig
    from miniverl.schemas.trajectory import TRAJECTORY_SCHEMA_VERSION, SpanType
    from miniverl.trainer import OPDTrainer
    from miniverl.trajectory.io import read_trajectories

    config = RunConfig.model_validate(
        {
            "run": {
                "name": "agentic-grpo",
                "mode": "rl",
                "seed": 31,
                "output_dir": str(tmp_path / "runs"),
            },
            "models": {
                "backend": "toy",
                "device": "cpu",
                "student": {
                    "model_id": "toy-agent",
                    "lora": {"enabled": False},
                    "toy": {
                        "hidden_size": 16,
                        "num_layers": 1,
                        "num_heads": 2,
                        "intermediate_size": 32,
                        "max_position_embeddings": 1024,
                    },
                },
            },
            "source": {"kind": "environment"},
            "environment": {
                "name": "calculator",
                "train_tasks": 1,
                "eval_tasks": 1,
                "test_tasks": 0,
            },
            "rollout": {
                "backend": "hf_reference",
                "samples_per_prompt": 2,
                "temperature": 1.0,
                "max_turns": 1,
                "max_new_tokens_per_turn": 4,
                "max_total_tokens": 768,
                "record_logprobs": True,
            },
            "selection": {"selector": "all_model_tokens"},
            "loss": {
                "mode": "verl_rl_policy",
                "aggregation": "token-mean",
                "scale_by_temperature_squared": False,
                "chunk_size": 16,
            },
            "algorithm": {
                "name": "grpo",
                "implementation_version": ADVANTAGE_IMPLEMENTATION_VERSION,
            },
            "reward": {"enabled": True, "provider": "environment_verifier"},
            "train": {
                "cycles": 1,
                "rollouts_per_cycle": 1,
                "gradient_accumulation_steps": 2,
                "learning_rate": 0.001,
            },
            "memory": {"strategy": "resident"},
            "eval": {"enabled": False},
            "report": {"enabled": False},
        }
    )

    trainer = OPDTrainer.from_config(config, run_id="agentic-grpo")
    try:
        result = trainer.train()
    finally:
        trainer.close()

    trajectories = read_trajectories(result.run_dir / "trajectories.jsonl")
    assert len(trajectories) == 2
    assert all(item.schema_version == TRAJECTORY_SCHEMA_VERSION for item in trajectories)
    assert {(item.prompt_group_id, item.sample_index) for item in trajectories} == {
        (trajectories[0].prompt_group_id, 0),
        (trajectories[0].prompt_group_id, 1),
    }
    for trajectory in trajectories:
        old = trajectory.metadata["actor_rollout_log_probs"]
        assert len(old) == sum(trajectory.model_generated_mask)
        for span in trajectory.spans:
            if span.span_type in {SpanType.SYSTEM, SpanType.USER, SpanType.TOOL_RESULT}:
                assert not any(trajectory.model_generated_mask[span.start : span.end])
    rewards = [
        json.loads(line)
        for line in (result.run_dir / "rewards.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rewards) == 2
    assert {row["provider"]["name"] for row in rewards} == {"environment:calculator"}
    assert result.global_step == 1


def test_ppo_executes_actor_critic_gae_and_checkpoints_both_roles(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:  # type: ignore[no-untyped-def]
    import torch

    from miniverl.config import RunConfig
    from miniverl.trainer import OPDTrainer
    from miniverl.training.checkpoint import validate_checkpoint

    train = tmp_path / "ppo.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "prompt": "Choose one token.",
                    "data_source": "unit",
                    "reward_model": {"style": "exact", "ground_truth": "unused"},
                    "extra_info": {"ground_truth": "unused"},
                }
            ]
        ),
        train,
    )
    payload = {
        "run": {"name": "ppo", "mode": "rl", "seed": 23, "output_dir": str(tmp_path / "runs")},
        "models": {
            "backend": "toy",
            "device": "cpu",
            "student": {
                "model_id": "toy-ppo",
                "lora": {"enabled": False},
                "toy": {
                    "hidden_size": 16,
                    "num_layers": 1,
                    "num_heads": 2,
                    "intermediate_size": 32,
                    "max_position_embeddings": 128,
                },
            },
        },
        "source": {
            "kind": "verl_parquet",
            "train_files": [str(train)],
            "allow_plain_string_prompts": True,
            "shuffle": False,
            "use_task_rewards": True,
        },
        "rollout": {
            "backend": "hf_cached",
            "samples_per_prompt": 2,
            "temperature": 1.0,
            "max_turns": 1,
            "max_new_tokens_per_turn": 3,
            "max_total_tokens": 64,
            "record_logprobs": True,
        },
        "selection": {"selector": "all_model_tokens"},
        "loss": {
            "mode": "verl_rl_policy",
            "aggregation": "token-mean",
            "scale_by_temperature_squared": False,
            "chunk_size": 16,
        },
        "algorithm": {
            "name": "ppo",
            "implementation_version": ADVANTAGE_IMPLEMENTATION_VERSION,
            "gamma": 0.9,
            "lam": 0.8,
            "actor_ppo_epochs": 2,
        },
        "critic": {"enabled": True, "learning_rate": 0.002, "ppo_epochs": 2},
        "reward": {"enabled": True, "provider": "python_api"},
        "train": {
            "cycles": 1,
            "rollouts_per_cycle": 1,
            "gradient_accumulation_steps": 2,
            "trajectory_batch_size": 2,
            "learning_rate": 0.001,
        },
        "memory": {"strategy": "resident", "oom_retries": 1},
        "eval": {"enabled": False},
        "report": {"enabled": False},
    }
    config = RunConfig.model_validate(payload)
    trainer = OPDTrainer.from_config(config, run_id="ppo-e2e", reward_provider=_SampleIndexReward())
    assert trainer.critic is not None
    original_critic_gradients = trainer._ppo_runtime.compute_group_gradients
    attempts = 0

    def one_oom_then_succeed(group, *, physical_cap=None):  # type: ignore[no-untyped-def]
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("CUDA out of memory (fault injection)")
        return original_critic_gradients(group, physical_cap=physical_cap)

    monkeypatch.setattr(trainer._ppo_runtime, "compute_group_gradients", one_oom_then_succeed)
    actor_before = _trainable_state(trainer)
    critic_before = _critic_state(trainer)
    try:
        result = trainer.train()
        actor_after = _trainable_state(trainer)
        critic_after = _critic_state(trainer)
        checkpoint = validate_checkpoint(result.run_dir / "checkpoints" / "final")
    finally:
        trainer.close()

    assert result.final_metrics["verl_rl"]["algorithm"] == "ppo"
    assert result.final_metrics["critic"]["value_loss"] >= 0.0
    assert result.actor_update_count == 2
    assert result.critic_update_count == 2
    assert result.final_metrics["critic"]["update_count"] == 2
    assert trainer._ppo_runtime.microbatch_size_history == [2]
    assert any(not torch.equal(actor_before[name], actor_after[name]) for name in actor_before)
    assert any(not torch.equal(critic_before[name], critic_after[name]) for name in critic_before)
    assert "critic.safetensors" in checkpoint.manifest["files"]
    assert "critic-optimizer.safetensors" in checkpoint.manifest["files"]


def test_ppo_checkpoint_resume_restores_actor_critic_optimizers_and_versions(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import torch

    from miniverl.config import RunConfig
    from miniverl.trainer import OPDTrainer
    from miniverl.utils.seeding import seed_everything

    train = tmp_path / "ppo-resume.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "prompt": f"Choose one token for row {index}.",
                    "data_source": "unit",
                    "reward_model": {"style": "exact", "ground_truth": "unused"},
                    "extra_info": {"ground_truth": "unused"},
                }
                for index in range(2)
            ]
        ),
        train,
    )
    config = RunConfig.model_validate(
        {
            "run": {
                "name": "ppo-resume",
                "mode": "rl",
                "seed": 29,
                "output_dir": str(tmp_path / "runs"),
            },
            "models": {
                "backend": "toy",
                "device": "cpu",
                "student": {
                    "model_id": "toy-ppo-resume",
                    "lora": {"enabled": False},
                    "toy": {
                        "hidden_size": 16,
                        "num_layers": 1,
                        "num_heads": 2,
                        "intermediate_size": 32,
                        "max_position_embeddings": 128,
                    },
                },
            },
            "source": {
                "kind": "verl_parquet",
                "train_files": [str(train)],
                "allow_plain_string_prompts": True,
                "shuffle": False,
                "use_task_rewards": True,
            },
            "rollout": {
                "backend": "hf_cached",
                "samples_per_prompt": 2,
                "temperature": 1.0,
                "max_turns": 1,
                "max_new_tokens_per_turn": 3,
                "max_total_tokens": 64,
                "record_logprobs": True,
            },
            "selection": {"selector": "all_model_tokens"},
            "loss": {
                "mode": "verl_rl_policy",
                "aggregation": "token-mean",
                "scale_by_temperature_squared": False,
                "chunk_size": 16,
            },
            "algorithm": {
                "name": "ppo",
                "implementation_version": ADVANTAGE_IMPLEMENTATION_VERSION,
                "gamma": 0.9,
                "lam": 0.8,
            },
            "critic": {"enabled": True, "learning_rate": 0.002},
            "reward": {"enabled": True, "provider": "python_api"},
            "train": {
                "cycles": 2,
                "rollouts_per_cycle": 1,
                "gradient_accumulation_steps": 2,
                "learning_rate": 0.001,
            },
            "memory": {"strategy": "resident"},
            "eval": {"enabled": False},
            "report": {"enabled": False},
        }
    )

    seed_everything(29, deterministic=True)
    reference = OPDTrainer.from_config(
        config, run_id="ppo-reference", reward_provider=_SampleIndexReward()
    )
    try:
        reference_result = reference.train()
        reference_actor = _trainable_state(reference)
        reference_critic = _critic_state(reference)
        reference_critic_updates = reference.critic_update_count
    finally:
        reference.close()

    seed_everything(29, deterministic=True)
    interrupted = OPDTrainer.from_config(
        config, run_id="ppo-interrupted", reward_provider=_SampleIndexReward()
    )
    interrupted.cycle = 0
    interrupted._run_cycle()
    interrupted.save_checkpoint(name="interrupt")
    interrupted_root = interrupted.paths.root
    interrupted.close()

    resumed = OPDTrainer.from_config(
        config, resume=interrupted_root, reward_provider=_SampleIndexReward()
    )
    try:
        resumed_result = resumed.train()
        resumed_actor = _trainable_state(resumed)
        resumed_critic = _critic_state(resumed)
        resumed_critic_updates = resumed.critic_update_count
    finally:
        resumed.close()

    assert resumed_result.global_step == reference_result.global_step == 2
    assert resumed_critic_updates == reference_critic_updates == 2
    for name in reference_actor:
        assert torch.equal(reference_actor[name], resumed_actor[name]), name
    for name in reference_critic:
        assert torch.equal(reference_critic[name], resumed_critic[name]), name


def test_rl_checkpoint_resume_matches_uninterrupted_policy_and_group_progress(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import torch

    from miniverl.config import RunConfig
    from miniverl.trainer import OPDTrainer
    from miniverl.trajectory.io import read_trajectories
    from miniverl.utils.seeding import seed_everything

    train = tmp_path / "resume.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "prompt": f"Choose one token for row {index}.",
                    "data_source": "unit",
                    "reward_model": {"style": "exact", "ground_truth": "unused"},
                    "extra_info": {"ground_truth": "unused"},
                }
                for index in range(2)
            ]
        ),
        train,
    )
    config = RunConfig.model_validate(
        {
            "run": {
                "name": "rl-resume",
                "mode": "rl",
                "seed": 43,
                "output_dir": str(tmp_path / "runs"),
            },
            "models": {
                "backend": "toy",
                "device": "cpu",
                "student": {
                    "model_id": "toy-resume-actor",
                    "lora": {"enabled": False},
                    "toy": {
                        "hidden_size": 16,
                        "num_layers": 1,
                        "num_heads": 2,
                        "intermediate_size": 32,
                        "max_position_embeddings": 128,
                    },
                },
            },
            "source": {
                "kind": "verl_parquet",
                "train_files": [str(train)],
                "allow_plain_string_prompts": True,
                "shuffle": False,
                "use_task_rewards": True,
            },
            "rollout": {
                "backend": "hf_cached",
                "samples_per_prompt": 2,
                "temperature": 1.0,
                "max_turns": 1,
                "max_new_tokens_per_turn": 3,
                "max_total_tokens": 64,
                "record_logprobs": True,
            },
            "selection": {"selector": "all_model_tokens"},
            "loss": {
                "mode": "verl_rl_policy",
                "aggregation": "token-mean",
                "scale_by_temperature_squared": False,
                "chunk_size": 16,
            },
            "algorithm": {
                "name": "grpo",
                "implementation_version": ADVANTAGE_IMPLEMENTATION_VERSION,
            },
            "reward": {"enabled": True, "provider": "python_api"},
            "train": {
                "cycles": 2,
                "rollouts_per_cycle": 1,
                "gradient_accumulation_steps": 2,
                "learning_rate": 0.001,
            },
            "memory": {"strategy": "resident"},
            "eval": {"enabled": False},
            "report": {"enabled": False},
        }
    )

    seed_everything(43, deterministic=True)
    reference = OPDTrainer.from_config(
        config, run_id="rl-reference", reward_provider=_SampleIndexReward()
    )
    try:
        reference_result = reference.train()
        reference_state = _trainable_state(reference)
    finally:
        reference.close()

    seed_everything(43, deterministic=True)
    interrupted = OPDTrainer.from_config(
        config, run_id="rl-interrupted", reward_provider=_SampleIndexReward()
    )
    interrupted.cycle = 0
    interrupted._run_cycle()
    checkpoint = interrupted.save_checkpoint(name="interrupt")
    interrupted_root = interrupted.paths.root
    interrupted.close()

    class _DifferentReward(_SampleIndexReward):
        identity = RewardProviderIdentity(
            name="different",
            version="v1",
            config_digest="2" * 64,
            deterministic=True,
        )

    from miniverl.errors import ConfigError

    with pytest.raises(ConfigError, match="reward provider identity"):
        OPDTrainer.from_config(
            config,
            resume=interrupted_root,
            reward_provider=_DifferentReward(),
        )

    resumed = OPDTrainer.from_config(
        config,
        resume=interrupted_root,
        reward_provider=_SampleIndexReward(),
    )
    try:
        resumed_result = resumed.train()
        resumed_state = _trainable_state(resumed)
    finally:
        resumed.close()

    assert checkpoint.is_dir()
    assert resumed_result.global_step == reference_result.global_step == 2
    assert resumed_result.policy_version == reference_result.policy_version == 2
    resumed_rows = read_trajectories(resumed_result.run_dir / "trajectories.jsonl")
    reference_rows = read_trajectories(reference_result.run_dir / "trajectories.jsonl")
    assert [(row.prompt_group_id, row.sample_index) for row in resumed_rows] == [
        (row.prompt_group_id, row.sample_index) for row in reference_rows
    ]
    assert len(resumed_rows) == len(reference_rows) == 4
    assert resumed_state.keys() == reference_state.keys()
    for name in resumed_state:
        assert torch.equal(resumed_state[name], reference_state[name]), name
