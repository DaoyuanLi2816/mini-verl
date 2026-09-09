from __future__ import annotations

import math

import pytest

from miniverl.errors import ConfigError
from miniverl.rewards import (
    ADVANTAGE_COMPOSER_VERSION,
    AdvantageComposer,
    AdvantageMode,
    EnvironmentVerifierRewardProvider,
    ExactAnswerRewardProvider,
    RewardProviderIdentity,
    RewardRequest,
    RewardResult,
    RewardStatus,
    TargetLengthRewardProvider,
    WeightedRewardProvider,
    score_reward_requests,
)


def _request(*, response: str = "42", reward_model=None) -> RewardRequest:  # type: ignore[no-untyped-def]
    return RewardRequest.create(
        trajectory_id="t0",
        prompt_group_id="g0",
        sample_index=0,
        samples_per_prompt=2,
        row_digest="a" * 64,
        response_text=response,
        reward_model=reward_model or {"style": "exact", "ground_truth": "42"},
        ground_truth="42",
        data_source="unit",
    )


def test_reward_request_digest_binds_all_deterministic_inputs() -> None:
    first = _request()
    same = _request()
    changed = _request(response="41")

    assert first.input_digest == same.input_digest
    assert first.input_digest != changed.input_digest
    assert len(first.input_digest) == 64


def test_exact_answer_provider_records_success_and_failure_without_code_loading() -> None:
    provider = ExactAnswerRewardProvider()

    solved = provider.score(_request())
    failed = provider.score(_request(response="41"))

    assert solved.status is RewardStatus.OK
    assert solved.raw_reward == 1.0
    assert solved.components[0].name == "exact_match"
    assert failed.status is RewardStatus.OK
    assert failed.raw_reward == 0.0
    assert failed.failure_category == "answer_mismatch"
    assert solved.provider.name == "builtin_exact_answer"
    assert solved.deterministic is True


def test_exact_answer_provider_returns_error_not_zero_for_invalid_metadata() -> None:
    provider = ExactAnswerRewardProvider()
    request = _request(reward_model={"style": "python", "module": "untrusted.py"})

    result = provider.score(request)

    assert result.status is RewardStatus.ERROR
    assert result.raw_reward is None
    assert result.failure_category == "invalid_reward_metadata"


def test_target_length_provider_scores_declared_character_distance() -> None:
    provider = TargetLengthRewardProvider()
    exact = provider.score(
        _request(response="four", reward_model={"style": "target_length", "characters": 4})
    )
    near = provider.score(
        _request(response="three", reward_model={"style": "target_length", "characters": 4})
    )
    invalid = provider.score(
        _request(response="four", reward_model={"style": "target_length", "characters": 0})
    )

    assert exact.status is RewardStatus.OK
    assert exact.raw_reward == 1.0
    assert exact.components[0].name == "target_length"
    assert near.raw_reward == pytest.approx(0.75)
    assert invalid.status is RewardStatus.ERROR
    assert invalid.raw_reward is None
    assert invalid.failure_category == "invalid_reward_metadata"


def test_environment_verifier_adapter_preserves_components() -> None:
    from miniverl.environments.base import FailureCategory, VerificationResult

    class Verifier:
        name = "unit_env"
        verifier_version = "v2"

        def __init__(self) -> None:
            self.params = {"mode": "strict"}

        def verify(self, answer: str) -> VerificationResult:
            return VerificationResult(
                solved=answer == "yes",
                reward=0.75,
                expected="yes",
                predicted=answer,
                failure_category=FailureCategory.SOLVED,
                detail="unit verifier",
            )

    result = EnvironmentVerifierRewardProvider(Verifier()).score(_request(response="yes"))

    assert result.status is RewardStatus.OK
    assert result.raw_reward == 0.75
    assert result.components[0].name == "environment_verifier"
    assert result.provider.name == "environment:unit_env"


@pytest.mark.parametrize("reward", [float("nan"), float("inf"), float("-inf")])
def test_recorded_environment_verifier_rejects_non_finite_rewards(reward: float) -> None:
    class Verifier:
        name = "unit_env"
        verifier_version = "v2"

        def __init__(self) -> None:
            self.params: dict[str, object] = {}

    request = _request().model_copy(
        update={
            "reward_model": {
                "style": "recorded_environment_verifier",
                "reward": reward,
            }
        }
    )
    result = EnvironmentVerifierRewardProvider(Verifier()).score(request)

    assert result.status is RewardStatus.ERROR
    assert result.raw_reward is None
    assert result.failure_category == "invalid_recorded_verification"


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (AdvantageMode.NONE, [0.0, 0.0, 0.0]),
        (AdvantageMode.RAW, [1.0, 2.0, 4.0]),
        (AdvantageMode.GROUP_CENTER, [-4 / 3, -1 / 3, 5 / 3]),
        (
            AdvantageMode.GROUP_STANDARDIZE,
            [
                (-4 / 3) / math.sqrt(14 / 9),
                (-1 / 3) / math.sqrt(14 / 9),
                (5 / 3) / math.sqrt(14 / 9),
            ],
        ),
        (AdvantageMode.LEAVE_ONE_OUT, [-2.0, -0.5, 2.5]),
    ],
)
def test_advantage_composer_modes_are_explicit(
    mode: AdvantageMode,
    expected: list[float],
) -> None:
    composer = AdvantageComposer(
        mode=mode,
        distillation_coef=0.25,
        task_reward_coef=2.0,
    )

    result = composer.compose_group([1.0, 2.0, 4.0])

    assert result.task_advantages == pytest.approx(expected)
    assert result.distillation_coef == 0.25
    assert result.task_reward_coef == 2.0
    assert result.implementation_version == ADVANTAGE_COMPOSER_VERSION


@pytest.mark.parametrize(
    "mode",
    [AdvantageMode.GROUP_CENTER, AdvantageMode.GROUP_STANDARDIZE, AdvantageMode.LEAVE_ONE_OUT],
)
def test_zero_variance_group_is_defined_as_zero(mode: AdvantageMode) -> None:
    result = AdvantageComposer(mode=mode).compose_group([0.5, 0.5, 0.5])

    assert result.task_advantages == [0.0, 0.0, 0.0]
    assert result.zero_variance is True


def test_composer_expands_task_scalar_only_to_selected_assistant_tokens() -> None:
    composer = AdvantageComposer(mode=AdvantageMode.RAW)

    expanded = composer.expand_task_advantage(2.0, [False, True, True, False])

    assert expanded == [0.0, 2.0, 2.0, 0.0]


def test_weighted_reward_preserves_named_components_and_identity() -> None:
    provider = WeightedRewardProvider(
        [
            ("correctness", 2.0, ExactAnswerRewardProvider()),
            ("format", -0.25, ExactAnswerRewardProvider(strip_whitespace=False)),
        ]
    )

    result = provider.score(_request())

    assert result.status is RewardStatus.OK
    assert result.raw_reward == pytest.approx(1.75)
    assert [(item.name, item.value) for item in result.components] == [
        ("correctness", 2.0),
        ("format", -0.25),
    ]
    assert provider.identity.name == "weighted_composite"
    assert len(provider.identity.digest) == 64


@pytest.mark.parametrize("mismatch", ["trajectory", "provider"])
def test_batch_reward_results_must_match_request_order_and_provider(mismatch: str) -> None:
    request = _request()

    class MisboundProvider:
        identity = RewardProviderIdentity(
            name="misbound",
            version="v1",
            config_digest="b" * 64,
            deterministic=True,
        )

        def score_batch(self, requests):  # type: ignore[no-untyped-def]
            return [
                RewardResult(
                    trajectory_id="another-trajectory"
                    if mismatch == "trajectory"
                    else item.trajectory_id,
                    prompt_group_id=item.prompt_group_id,
                    sample_index=item.sample_index,
                    samples_per_prompt=item.samples_per_prompt,
                    provider=self.identity
                    if mismatch == "trajectory"
                    else self.identity.model_copy(update={"config_digest": "c" * 64}),
                    input_digest=item.input_digest,
                    raw_reward=1.0,
                    status=RewardStatus.OK,
                    duration_ms=0.0,
                    deterministic=True,
                )
                for item in requests
            ]

        def score(self, request):  # type: ignore[no-untyped-def]
            raise AssertionError("batch path expected")

    with pytest.raises(ConfigError, match="identity does not match"):
        score_reward_requests(MisboundProvider(), [request])


@pytest.mark.torch
@pytest.mark.parametrize("failure", [None, "timeout", "model_failure"])
def test_hf_reward_model_scores_batches_with_pinned_identity(monkeypatch, failure) -> None:
    import torch

    from miniverl.config import RewardModelConfig
    from miniverl.rewards import HFSequenceClassifierRewardProvider, RewardRequest

    class Tokenizer:
        def __call__(self, prompts, responses, **kwargs):  # type: ignore[no-untyped-def]
            assert kwargs["padding"] is True
            values = [
                [len(prompt), len(response)]
                for prompt, response in zip(prompts, responses, strict=True)
            ]
            return {"input_ids": torch.tensor(values)}

    class Model:
        def __call__(self, *, input_ids):  # type: ignore[no-untyped-def]
            from types import SimpleNamespace

            return SimpleNamespace(
                logits=torch.stack((-input_ids[:, 1], input_ids[:, 1]), dim=-1).float()
            )

        def to(self, _device):  # type: ignore[no-untyped-def]
            return self

    config = RewardModelConfig(
        model_id="org/reward-model",
        revision="a" * 40,
        batch_size=2,
    )
    provider = HFSequenceClassifierRewardProvider(
        config, tokenizer=Tokenizer(), model=Model(), device="cpu"
    )
    requests = [
        RewardRequest.create(
            trajectory_id=f"t{index}",
            prompt_group_id="g",
            sample_index=index,
            samples_per_prompt=2,
            row_digest="1" * 64,
            prompt_text="prompt",
            response_text="x" * (index + 1),
            reward_model={},
            ground_truth=None,
            data_source="unit",
        )
        for index in range(2)
    ]

    if failure == "timeout":
        times = iter([0.0, 0.0, 10000.0])
        monkeypatch.setattr("miniverl.rewards.providers.time.perf_counter", lambda: next(times))
        with pytest.raises(TimeoutError, match="exceeded"):
            provider.score_batch(requests)
        return
    if failure == "model_failure":

        def fail(**kwargs):
            raise RuntimeError("injected reward forward failure")

        provider.model = fail
        with pytest.raises(ConfigError, match="injected reward"):
            score_reward_requests(provider, requests)
        return
    results = provider.score_batch(requests)
    provider.to_device("cpu")
    provider.release()

    assert len(results) == 2
    assert results[1].raw_reward > results[0].raw_reward
    assert provider.identity.name == "hf_sequence_classifier"
    assert provider.identity.deterministic is True
    assert all(result.provider == provider.identity for result in results)
    assert provider.device == "cpu"


def test_hf_reward_model_config_is_explicit_and_provider_bound() -> None:
    from miniverl.config import RunConfig

    payload = {
        "run": {"mode": "rl"},
        "models": {
            "backend": "toy",
            "device": "cpu",
            "student": {"model_id": "actor", "lora": {"enabled": False}},
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
            "implementation_version": "verl-v0.9-advantages-v1",
        },
        "reward": {
            "enabled": True,
            "provider": "hf_sequence_classifier",
            "model": {"model_id": "org/rm", "revision": "b" * 40},
        },
        "train": {"cycles": 1, "rollouts_per_cycle": 1},
        "eval": {"enabled": False},
    }

    config = RunConfig.model_validate(payload)

    assert config.reward.model is not None
    assert config.reward.model.revision == "b" * 40


def test_hf_reward_model_requires_immutable_revisions() -> None:
    from pydantic import ValidationError

    from miniverl.config import RewardModelConfig

    with pytest.raises(ValidationError, match="revision"):
        RewardModelConfig(model_id="org/rm", revision="main")
    with pytest.raises(ValidationError, match="tokenizer_revision"):
        RewardModelConfig(
            model_id="org/rm",
            revision="a" * 40,
            tokenizer_revision="latest",
        )
