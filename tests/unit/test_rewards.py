from __future__ import annotations

import math

import pytest

from miniverl.rewards import (
    ADVANTAGE_COMPOSER_VERSION,
    AdvantageComposer,
    AdvantageMode,
    EnvironmentVerifierRewardProvider,
    ExactAnswerRewardProvider,
    RewardRequest,
    RewardStatus,
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
