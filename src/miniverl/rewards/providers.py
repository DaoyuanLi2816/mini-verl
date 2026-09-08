"""Built-in deterministic reward providers; no artifact-driven code loading."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from miniverl.errors import ConfigError
from miniverl.rewards.models import (
    RewardComponent,
    RewardProviderIdentity,
    RewardRequest,
    RewardResult,
    RewardStatus,
)

__all__ = [
    "BatchRewardProvider",
    "EnvironmentVerifierRewardProvider",
    "ExactAnswerRewardProvider",
    "TargetLengthRewardProvider",
    "RewardProvider",
    "WeightedRewardProvider",
    "score_reward_requests",
]


def _config_digest(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


@runtime_checkable
class RewardProvider(Protocol):
    @property
    def identity(self) -> RewardProviderIdentity: ...

    def score(self, request: RewardRequest) -> RewardResult: ...


@runtime_checkable
class BatchRewardProvider(Protocol):
    @property
    def identity(self) -> RewardProviderIdentity: ...

    def score_batch(self, requests: Sequence[RewardRequest]) -> Sequence[RewardResult]: ...


def score_reward_requests(
    provider: RewardProvider,
    requests: Sequence[RewardRequest],
) -> list[RewardResult]:
    """Score in order and verify that each result is bound to its request."""
    try:
        if isinstance(provider, BatchRewardProvider):
            results = list(provider.score_batch(requests))
        else:
            results = [provider.score(request) for request in requests]
    except Exception as exc:
        raise ConfigError(f"reward provider {provider.identity.name!r} raised: {exc}") from exc
    if len(results) != len(requests):
        raise ConfigError(
            f"reward provider returned {len(results)} results for {len(requests)} requests"
        )
    for request, result in zip(requests, results, strict=True):
        expected = (
            request.trajectory_id,
            request.prompt_group_id,
            request.sample_index,
            request.samples_per_prompt,
            request.input_digest,
            provider.identity,
        )
        actual = (
            result.trajectory_id,
            result.prompt_group_id,
            result.sample_index,
            result.samples_per_prompt,
            result.input_digest,
            result.provider,
        )
        if actual != expected:
            raise ConfigError(
                "reward result identity does not match its request/provider; "
                f"trajectory={request.trajectory_id!r}"
            )
    return results


class WeightedRewardProvider:
    """Deterministically combine trusted providers with explicit scalar weights."""

    def __init__(self, components: Sequence[tuple[str, float, RewardProvider]]) -> None:
        if not components:
            raise ValueError("weighted reward requires at least one component")
        names = [name for name, _, _ in components]
        if len(set(names)) != len(names) or any(not name for name in names):
            raise ValueError("weighted reward component names must be unique and non-empty")
        if any(not math.isfinite(weight) for _, weight, _ in components):
            raise ValueError("weighted reward component weights must be finite")
        self.components = tuple(components)
        config = [
            {
                "name": name,
                "weight": weight,
                "provider_identity": provider.identity.model_dump(mode="json"),
            }
            for name, weight, provider in self.components
        ]
        self._identity = RewardProviderIdentity(
            name="weighted_composite",
            version="miniverl-weighted-reward-v1",
            config_digest=_config_digest(config),
            package_name="miniverl",
            deterministic=all(provider.identity.deterministic for _, _, provider in components),
        )

    @property
    def identity(self) -> RewardProviderIdentity:
        return self._identity

    def score(self, request: RewardRequest) -> RewardResult:
        started = time.perf_counter()
        values: list[tuple[str, float, float, str]] = []
        for name, weight, provider in self.components:
            child = score_reward_requests(provider, [request])[0]
            if child.status is not RewardStatus.OK or child.raw_reward is None:
                return RewardResult(
                    trajectory_id=request.trajectory_id,
                    prompt_group_id=request.prompt_group_id,
                    sample_index=request.sample_index,
                    samples_per_prompt=request.samples_per_prompt,
                    provider=self.identity,
                    input_digest=request.input_digest,
                    raw_reward=None,
                    status=RewardStatus.ERROR,
                    failure_category=f"component:{name}:{child.failure_category or child.status.value}",
                    detail=child.detail,
                    duration_ms=max((time.perf_counter() - started) * 1000.0, 0.0),
                    deterministic=self.identity.deterministic,
                )
            values.append((name, weight, float(child.raw_reward), child.provider.name))
        total = sum(weight * value for _, weight, value, _ in values)
        return RewardResult(
            trajectory_id=request.trajectory_id,
            prompt_group_id=request.prompt_group_id,
            sample_index=request.sample_index,
            samples_per_prompt=request.samples_per_prompt,
            provider=self.identity,
            input_digest=request.input_digest,
            raw_reward=total,
            components=tuple(
                RewardComponent(
                    name=name,
                    value=weight * value,
                    detail=f"weight={weight}; provider={provider_name}",
                )
                for name, weight, value, provider_name in values
            ),
            status=RewardStatus.OK,
            duration_ms=max((time.perf_counter() - started) * 1000.0, 0.0),
            deterministic=self.identity.deterministic,
        )

    def score_batch(self, requests: Sequence[RewardRequest]) -> Sequence[RewardResult]:
        return [self.score(request) for request in requests]


class ExactAnswerRewardProvider:
    """Strict string equality over declared Parquet ground truth."""

    def __init__(self, *, strip_whitespace: bool = True) -> None:
        self.strip_whitespace = strip_whitespace
        self._identity = RewardProviderIdentity(
            name="builtin_exact_answer",
            version="miniverl-exact-answer-v1",
            config_digest=_config_digest({"strip_whitespace": strip_whitespace}),
            package_name="miniverl",
            deterministic=True,
        )

    @property
    def identity(self) -> RewardProviderIdentity:
        return self._identity

    def score(self, request: RewardRequest) -> RewardResult:
        started = time.perf_counter()
        metadata = request.reward_model
        valid = isinstance(metadata, dict) and metadata.get("style") in {"exact", "rule"}
        declared = metadata.get("ground_truth") if valid else None
        if declared is None and valid:
            declared = request.ground_truth
        if not valid or not isinstance(declared, (str, int, float, bool)):
            return self._result(
                request,
                started=started,
                status=RewardStatus.ERROR,
                raw_reward=None,
                failure_category="invalid_reward_metadata",
                detail=(
                    "reward_model must declare style exact/rule and a scalar ground_truth; "
                    "arbitrary Python reward metadata is not executable"
                ),
            )
        expected = str(declared)
        predicted = request.response_text
        if self.strip_whitespace:
            expected = expected.strip()
            predicted = predicted.strip()
        solved = predicted == expected
        return self._result(
            request,
            started=started,
            status=RewardStatus.OK,
            raw_reward=1.0 if solved else 0.0,
            components=(
                RewardComponent(
                    name="exact_match",
                    value=1.0 if solved else 0.0,
                    detail="deterministic normalized string equality",
                ),
            ),
            failure_category=None if solved else "answer_mismatch",
            detail=None if solved else f"expected {expected!r}, received {predicted!r}",
        )

    def _result(
        self,
        request: RewardRequest,
        *,
        started: float,
        status: RewardStatus,
        raw_reward: float | None,
        components: tuple[RewardComponent, ...] = (),
        failure_category: str | None,
        detail: str | None,
    ) -> RewardResult:
        return RewardResult(
            trajectory_id=request.trajectory_id,
            prompt_group_id=request.prompt_group_id,
            sample_index=request.sample_index,
            samples_per_prompt=request.samples_per_prompt,
            provider=self.identity,
            input_digest=request.input_digest,
            raw_reward=raw_reward,
            components=components,
            status=status,
            failure_category=failure_category,
            detail=detail,
            duration_ms=max((time.perf_counter() - started) * 1000.0, 0.0),
            deterministic=True,
        )


class TargetLengthRewardProvider:
    """Score closeness to an explicit response-length target without loading code."""

    def __init__(self, *, strip_whitespace: bool = True) -> None:
        self.strip_whitespace = strip_whitespace
        self._identity = RewardProviderIdentity(
            name="builtin_target_length",
            version="miniverl-target-length-v1",
            config_digest=_config_digest({"strip_whitespace": strip_whitespace}),
            package_name="miniverl",
            deterministic=True,
        )

    @property
    def identity(self) -> RewardProviderIdentity:
        return self._identity

    def score(self, request: RewardRequest) -> RewardResult:
        started = time.perf_counter()
        metadata = request.reward_model
        target = metadata.get("characters") if isinstance(metadata, dict) else None
        valid = (
            isinstance(metadata, dict)
            and metadata.get("style") == "target_length"
            and isinstance(target, int)
            and not isinstance(target, bool)
            and 1 <= target <= 1_000_000
        )
        if not valid:
            return RewardResult(
                trajectory_id=request.trajectory_id,
                prompt_group_id=request.prompt_group_id,
                sample_index=request.sample_index,
                samples_per_prompt=request.samples_per_prompt,
                provider=self.identity,
                input_digest=request.input_digest,
                raw_reward=None,
                status=RewardStatus.ERROR,
                failure_category="invalid_reward_metadata",
                detail=(
                    "reward_model must declare style target_length and an integer "
                    "characters target in [1, 1000000]"
                ),
                duration_ms=max((time.perf_counter() - started) * 1000.0, 0.0),
                deterministic=True,
            )
        assert isinstance(target, int)
        response = request.response_text.strip() if self.strip_whitespace else request.response_text
        distance = abs(len(response) - target)
        score = max(0.0, 1.0 - distance / target)
        return RewardResult(
            trajectory_id=request.trajectory_id,
            prompt_group_id=request.prompt_group_id,
            sample_index=request.sample_index,
            samples_per_prompt=request.samples_per_prompt,
            provider=self.identity,
            input_digest=request.input_digest,
            raw_reward=score,
            components=(
                RewardComponent(
                    name="target_length",
                    value=score,
                    detail=f"observed_characters={len(response)}; target_characters={target}",
                ),
            ),
            status=RewardStatus.OK,
            duration_ms=max((time.perf_counter() - started) * 1000.0, 0.0),
            deterministic=True,
        )


class EnvironmentVerifierRewardProvider:
    """Adapter for the active deterministic ToolEnvironment verifier."""

    def __init__(self, environment: Any) -> None:
        self.environment = environment
        name = str(getattr(environment, "name", type(environment).__name__))
        version = str(getattr(environment, "verifier_version", "unspecified"))
        self._identity = RewardProviderIdentity(
            name=f"environment:{name}",
            version=version,
            config_digest=_config_digest(dict(getattr(environment, "params", {}))),
            package_name="miniverl",
            deterministic=True,
        )

    @property
    def identity(self) -> RewardProviderIdentity:
        return self._identity

    def score(self, request: RewardRequest) -> RewardResult:
        started = time.perf_counter()
        recorded = request.reward_model
        if isinstance(recorded, dict) and recorded.get("style") == "recorded_environment_verifier":
            reward = recorded.get("reward")
            if (
                not isinstance(reward, (int, float))
                or isinstance(reward, bool)
                or not math.isfinite(float(reward))
            ):
                return RewardResult(
                    trajectory_id=request.trajectory_id,
                    prompt_group_id=request.prompt_group_id,
                    sample_index=request.sample_index,
                    samples_per_prompt=request.samples_per_prompt,
                    provider=self.identity,
                    input_digest=request.input_digest,
                    raw_reward=None,
                    status=RewardStatus.ERROR,
                    failure_category="invalid_recorded_verification",
                    detail="recorded environment verification lacks a finite numeric reward",
                    duration_ms=max((time.perf_counter() - started) * 1000.0, 0.0),
                    deterministic=True,
                )
            return RewardResult(
                trajectory_id=request.trajectory_id,
                prompt_group_id=request.prompt_group_id,
                sample_index=request.sample_index,
                samples_per_prompt=request.samples_per_prompt,
                provider=self.identity,
                input_digest=request.input_digest,
                raw_reward=float(reward),
                components=(
                    RewardComponent(
                        name="environment_verifier",
                        value=float(reward),
                        detail=str(recorded.get("detail") or "recorded during agent rollout"),
                    ),
                ),
                status=RewardStatus.OK,
                failure_category=(
                    str(recorded["failure_category"])
                    if recorded.get("failure_category") is not None
                    else None
                ),
                detail=str(recorded.get("detail") or "") or None,
                duration_ms=max((time.perf_counter() - started) * 1000.0, 0.0),
                deterministic=True,
            )
        try:
            verification = self.environment.verify(request.response_text)
        except Exception as exc:
            return RewardResult(
                trajectory_id=request.trajectory_id,
                prompt_group_id=request.prompt_group_id,
                sample_index=request.sample_index,
                samples_per_prompt=request.samples_per_prompt,
                provider=self.identity,
                input_digest=request.input_digest,
                raw_reward=None,
                status=RewardStatus.ERROR,
                failure_category="environment_verifier_error",
                detail=str(exc),
                duration_ms=max((time.perf_counter() - started) * 1000.0, 0.0),
                deterministic=True,
            )
        category = getattr(verification.failure_category, "value", verification.failure_category)
        return RewardResult(
            trajectory_id=request.trajectory_id,
            prompt_group_id=request.prompt_group_id,
            sample_index=request.sample_index,
            samples_per_prompt=request.samples_per_prompt,
            provider=self.identity,
            input_digest=request.input_digest,
            raw_reward=float(verification.reward),
            components=(
                RewardComponent(
                    name="environment_verifier",
                    value=float(verification.reward),
                    detail=verification.detail,
                ),
            ),
            status=RewardStatus.OK,
            failure_category=str(category) if category is not None else None,
            detail=verification.detail,
            duration_ms=max((time.perf_counter() - started) * 1000.0, 0.0),
            deterministic=True,
        )
