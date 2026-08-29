"""Built-in deterministic reward providers; no artifact-driven code loading."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Protocol, runtime_checkable

from miniverl.rewards.models import (
    RewardComponent,
    RewardProviderIdentity,
    RewardRequest,
    RewardResult,
    RewardStatus,
)

__all__ = [
    "EnvironmentVerifierRewardProvider",
    "ExactAnswerRewardProvider",
    "RewardProvider",
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
