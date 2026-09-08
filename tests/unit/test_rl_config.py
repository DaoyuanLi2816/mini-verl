from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from miniverl.algorithms.contract import ADVANTAGE_IMPLEMENTATION_VERSION
from miniverl.config import RunConfig


def rl_config() -> dict:  # type: ignore[type-arg]
    return {
        "run": {"name": "local-grpo", "mode": "rl", "seed": 7},
        "models": {
            "backend": "toy",
            "device": "cpu",
            "student": {"model_id": "student", "lora": {"enabled": False}},
        },
        "source": {
            "kind": "verl_parquet",
            "train_files": ["train.parquet"],
            "use_task_rewards": True,
        },
        "rollout": {
            "backend": "hf_cached",
            "samples_per_prompt": 2,
            "temperature": 1.0,
            "record_logprobs": True,
        },
        "selection": {"selector": "all_model_tokens"},
        "loss": {
            "mode": "verl_rl_policy",
            "aggregation": "token-mean",
            "scale_by_temperature_squared": False,
        },
        "algorithm": {
            "name": "grpo",
            "implementation_version": ADVANTAGE_IMPLEMENTATION_VERSION,
        },
        "reward": {"enabled": True, "provider": "exact_answer"},
        "train": {"cycles": 1, "rollouts_per_cycle": 1},
        "eval": {"enabled": False},
        "report": {"enabled": False},
    }


def test_teacher_free_grpo_config_round_trips() -> None:
    config = RunConfig.model_validate(rl_config())
    assert config.models.teacher is None
    assert config.algorithm.name.value == "grpo"
    assert RunConfig.model_validate(config.model_dump()).algorithm == config.algorithm


def test_parquet_rl_accepts_the_builtin_target_length_reward() -> None:
    payload = rl_config()
    payload["reward"]["provider"] = "target_length"

    assert RunConfig.model_validate(payload).reward.provider.value == "target_length"


@pytest.mark.parametrize("algorithm", ["grpo", "dr_grpo", "rloo"])
def test_group_estimators_require_real_grouped_rollouts(algorithm: str) -> None:
    payload = rl_config()
    payload["algorithm"]["name"] = algorithm
    payload["rollout"]["samples_per_prompt"] = 1
    with pytest.raises(ValidationError, match="requires samples_per_prompt > 1"):
        RunConfig.model_validate(payload)


def test_rl_rejects_silent_teacher_and_missing_reward() -> None:
    payload = rl_config()
    payload["models"]["teacher"] = {"model_id": "unused"}
    with pytest.raises(ValidationError, match="teacher-free"):
        RunConfig.model_validate(payload)

    payload = deepcopy(rl_config())
    payload["reward"]["enabled"] = False
    with pytest.raises(ValidationError, match="requires task rewards"):
        RunConfig.model_validate(payload)


def test_ppo_fails_closed_until_critic_lifecycle_is_complete() -> None:
    payload = rl_config()
    payload["algorithm"]["name"] = "ppo"
    with pytest.raises(ValidationError, match="critic checkpoint/runtime contract"):
        RunConfig.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("kl_coef", 0.1, "requires models.reference"),
        ("value_loss_coef", 0.7, "not-yet-executable PPO critic path"),
        ("cliprange_value", 0.2, "not-yet-executable PPO critic path"),
    ],
)
def test_rl_rejects_unwired_algorithm_controls(field: str, value: float, match: str) -> None:
    payload = rl_config()
    payload["algorithm"][field] = value
    with pytest.raises(ValidationError, match=match):
        RunConfig.model_validate(payload)


def test_rl_reference_kl_requires_an_explicit_shared_reference_role() -> None:
    payload = rl_config()
    payload["models"] = {
        "backend": "hf",
        "runtime": "shared_backbone",
        "device": "cpu",
        "student": {
            "model_id": "org/base",
            "lora": {"enabled": True},
        },
        "reference": {
            "model_id": "org/base",
            "adapter": {"path": "reference-adapter", "source": "local"},
        },
    }
    payload["algorithm"]["kl_coef"] = 0.01

    config = RunConfig.model_validate(payload)

    assert config.models.teacher is None
    assert config.models.reference is not None
    assert config.algorithm.kl_penalty == "kl"
