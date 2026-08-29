from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError

from miniverl.rewards import ADVANTAGE_COMPOSER_VERSION


def _config() -> dict:  # type: ignore[type-arg]
    return {
        "run": {
            "name": "rewarded-pg",
            "mode": "opd",
            "seed": 9,
            "execution_plan_digest": "d" * 64,
        },
        "models": {
            "backend": "toy",
            "device": "cpu",
            "student": {"model_id": "student", "lora": {"enabled": False}},
            "teacher": {"model_id": "teacher", "toy_pretrain_steps": 0},
        },
        "source": {
            "kind": "verl_parquet",
            "train_files": ["train.parquet"],
            "use_task_rewards": True,
        },
        "rollout": {
            "backend": "hf_cached",
            "samples_per_prompt": 2,
            "record_logprobs": True,
        },
        "selection": {"selector": "all_model_tokens"},
        "loss": {
            "mode": "verl_pg_k1_rewarded",
            "aggregation": "token-mean",
            "divergence": "reverse_kl",
            "temperature": 1.0,
            "scale_by_temperature_squared": False,
            "top_k": 1,
            "estimator_implementation_version": "verl-v0.8-pg-k1-v1",
            "advantage_composer_version": ADVANTAGE_COMPOSER_VERSION,
            "advantage_mode": "group_center",
            "distillation_coef": 1.0,
            "task_reward_coef": 0.5,
        },
        "reward": {
            "enabled": True,
            "provider": "exact_answer",
            "error_policy": "fail",
        },
        "train": {"cycles": 1, "rollouts_per_cycle": 1},
        "eval": {"enabled": False},
        "report": {"enabled": False},
    }


def test_rewarded_profile_config_requires_group_for_group_normalization() -> None:
    from miniverl.config import RunConfig

    payload = _config()
    payload["rollout"]["samples_per_prompt"] = 1

    with pytest.raises(ValidationError, match="requires samples_per_prompt > 1"):
        RunConfig.model_validate(payload)


def test_rewarded_profile_config_requires_provider_and_fail_closed_policy() -> None:
    from miniverl.config import RunConfig

    payload = _config()
    payload["reward"]["enabled"] = False

    with pytest.raises(ValidationError, match=r"requires reward\.enabled=true"):
        RunConfig.model_validate(payload)


def test_existing_pg_profile_still_rejects_task_rewards() -> None:
    from miniverl.config import RunConfig

    payload = _config()
    payload["loss"].update(
        {
            "mode": "verl_pg_k1",
            "advantage_composer_version": None,
            "advantage_mode": "none",
            "distillation_coef": 1.0,
            "task_reward_coef": 0.0,
        }
    )
    payload["reward"] = {"enabled": False}

    with pytest.raises(ValidationError, match="existing PG-k1 profile is reward-free"):
        RunConfig.model_validate(payload)


def test_other_objectives_cannot_silently_ignore_reward_settings() -> None:
    from miniverl.config import RunConfig

    payload = _config()
    payload["source"]["use_task_rewards"] = False
    payload["loss"] = {"mode": "bucketed_topk_tail"}

    with pytest.raises(ValidationError, match="reward and advantage settings apply only"):
        RunConfig.model_validate(payload)


def test_valid_rewarded_profile_round_trips() -> None:
    from miniverl.config import RunConfig

    config = RunConfig.model_validate(deepcopy(_config()))

    assert config.source.use_task_rewards is True
    assert config.reward.provider.value == "exact_answer"
    assert config.loss.task_reward_coef == 0.5


def test_python_api_provider_is_explicit_config_not_an_artifact_import() -> None:
    from miniverl.config import RunConfig

    payload = _config()
    payload["reward"]["provider"] = "python_api"
    config = RunConfig.model_validate(payload)

    assert config.reward.provider.value == "python_api"
    assert "module" not in config.reward.model_dump()


@pytest.mark.torch
def test_rewarded_pg_combines_distillation_and_task_components_without_changing_mask() -> None:
    import torch

    from miniverl.losses.verl_pg import rewarded_pg_k1_loss

    current = torch.log(torch.tensor([0.4, 0.5, 0.6]))
    old = torch.log(torch.tensor([0.3, 0.5, 0.7]))
    teacher = torch.log(torch.tensor([0.6, 0.4, 0.8]))
    mask = torch.tensor([1.0, 1.0, 0.0])
    result = rewarded_pg_k1_loss(
        current_log_probs=current,
        old_log_probs=old,
        teacher_log_probs=teacher,
        weights=mask,
        task_advantage=0.25,
        distillation_coef=0.5,
        task_reward_coef=2.0,
    )

    expected_distill = teacher - old
    expected_total = 0.5 * expected_distill + 2.0 * 0.25
    torch.testing.assert_close(result.distillation_advantages, expected_distill)
    torch.testing.assert_close(result.task_advantages, torch.full_like(old, 0.25))
    torch.testing.assert_close(result.advantages, expected_total)
    assert result.metrics["distillation_advantage_mean"] == pytest.approx(
        float((expected_distill[:2]).mean())
    )
    assert result.metrics["task_advantage_mean"] == pytest.approx(0.25)


@pytest.mark.torch
def test_changing_only_task_reward_reverses_update_direction() -> None:
    import torch

    from miniverl.losses.verl_pg import rewarded_pg_k1_loss

    gradients = []
    for task_advantage in (1.0, -1.0):
        current = torch.tensor([-1.0], requires_grad=True)
        result = rewarded_pg_k1_loss(
            current_log_probs=current,
            old_log_probs=torch.tensor([-1.0]),
            teacher_log_probs=torch.tensor([-1.0]),
            weights=torch.ones(1),
            task_advantage=task_advantage,
            distillation_coef=1.0,
            task_reward_coef=1.0,
        )
        result.loss.backward()
        gradients.append(float(current.grad.item()))

    assert gradients[0] < 0.0
    assert gradients[1] > 0.0


def test_trainer_scores_complete_reward_group_and_persists_provenance(tmp_path: Path) -> None:
    from miniverl.config import RunConfig
    from miniverl.rewards import ExactAnswerRewardProvider
    from miniverl.schemas.trajectory import (
        TRAJECTORY_SCHEMA_VERSION,
        Span,
        SpanType,
        TerminationReason,
        Trajectory,
        derive_grouped_trajectory_id,
    )
    from miniverl.training.trainer import OPDTrainer
    from miniverl.utils.runs import JsonlWriter, read_jsonl

    config = RunConfig.model_validate(_config())
    trainer = object.__new__(OPDTrainer)
    trainer.config = config
    trainer.reward_provider = ExactAnswerRewardProvider()
    trainer.reward_log = JsonlWriter(tmp_path / "rewards.jsonl")
    trainer.advantage_log = JsonlWriter(tmp_path / "advantages.jsonl")
    policy_digest = "a" * 64
    prompt_digest = "b" * 64
    row_digest = "c" * 64
    trajectories = []
    for sample_index, response in enumerate(("answer", "wrong")):
        seed = 10 + sample_index
        trajectory_id = derive_grouped_trajectory_id(
            prompt_group_id="g0",
            sample_index=sample_index,
            rollout_policy_identity_digest=policy_digest,
            generation_seed=seed,
        )
        trajectories.append(
            Trajectory(
                schema_version=TRAJECTORY_SCHEMA_VERSION,
                trajectory_id=trajectory_id,
                task_id=row_digest,
                environment="verl_parquet",
                token_ids=[1],
                attention_mask=[1],
                model_generated_mask=[True],
                critical_mask=[False],
                spans=[
                    Span(
                        span_type=SpanType.ASSISTANT_TEXT,
                        start=0,
                        end=1,
                        turn_id=0,
                        text=response,
                    )
                ],
                tokenizer_fingerprint="fp",
                model_id="student",
                prompt_group_id="g0",
                prompt_digest=prompt_digest,
                sample_index=sample_index,
                samples_per_prompt=2,
                generation_seed=seed,
                rollout_backend="hf_cached",
                rollout_policy_identity_digest=policy_digest,
                termination_reason=TerminationReason.EOS_WITHOUT_FINAL,
                generated_token_count=1,
                metadata={
                    "row_digest": row_digest,
                    "data_source": "unit",
                    "reward_model": {"style": "exact", "ground_truth": "answer"},
                    "extra_info": {},
                },
            )
        )

    trainer._score_task_rewards(trajectories)

    assert [item.metadata["task_advantage"] for item in trajectories] == [0.5, -0.5]
    assert [item.verification.reward for item in trajectories if item.verification] == [1.0, 0.0]
    rows = read_jsonl(tmp_path / "rewards.jsonl")
    assert [row["raw_reward"] for row in rows] == [1.0, 0.0]
    assert all(row["provider"]["version"] == "miniverl-exact-answer-v1" for row in rows)
    advantages = read_jsonl(tmp_path / "advantages.jsonl")
    assert [row["task_advantage"] for row in advantages] == [0.5, -0.5]
    assert all(row["implementation_version"] == ADVANTAGE_COMPOSER_VERSION for row in advantages)
