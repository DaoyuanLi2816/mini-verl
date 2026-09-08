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
    "HFSequenceClassifierRewardProvider",
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


class HFSequenceClassifierRewardProvider:
    """Deterministic, pinned local reward-model inference role."""

    def __init__(self, config: Any, *, tokenizer: Any, model: Any, device: str) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        identity_config = {
            "model_id": config.model_id,
            "revision": config.revision,
            "tokenizer_id": config.tokenizer_id or config.model_id,
            "tokenizer_revision": config.tokenizer_revision or config.revision,
            "positive_class_index": config.positive_class_index,
            "max_length": config.max_length,
            "batch_size": config.batch_size,
            "offload_between_phases": config.offload_between_phases,
        }
        self._identity = RewardProviderIdentity(
            name="hf_sequence_classifier",
            version="miniverl-hf-rm-v1",
            config_digest=_config_digest(identity_config),
            package_name="miniverl",
            deterministic=True,
        )

    @classmethod
    def load(
        cls,
        config: Any,
        *,
        device: str,
        local_files_only: bool = False,
    ) -> HFSequenceClassifierRewardProvider:
        """Load the exact model/tokenizer revisions without config-driven imports."""
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - dependency guard exercised at CLI level
            from miniverl.errors import MissingDependencyError

            raise MissingDependencyError(
                "transformers", "train", "the Hugging Face reward-model role"
            ) from exc
        tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_id or config.model_id,
            revision=config.tokenizer_revision or config.revision,
            trust_remote_code=config.trust_remote_code,
            local_files_only=local_files_only,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            config.model_id,
            revision=config.revision,
            trust_remote_code=config.trust_remote_code,
            local_files_only=local_files_only,
            dtype=(torch.float16 if device.startswith("cuda") else torch.float32),
        )
        model.eval()
        initial_device = "cpu" if config.offload_between_phases else device
        model.to(initial_device)
        return cls(config, tokenizer=tokenizer, model=model, device=initial_device)

    @property
    def identity(self) -> RewardProviderIdentity:
        return self._identity

    def score(self, request: RewardRequest) -> RewardResult:
        return self.score_batch([request])[0]

    def score_batch(self, requests: Sequence[RewardRequest]) -> Sequence[RewardResult]:
        import torch

        started = time.perf_counter()
        outputs: list[RewardResult] = []
        for start in range(0, len(requests), self.config.batch_size):
            if time.perf_counter() - started > self.config.timeout_seconds:
                raise TimeoutError(
                    f"reward-model scoring exceeded {self.config.timeout_seconds:g} seconds"
                )
            batch = list(requests[start : start + self.config.batch_size])
            encoded = self.tokenizer(
                [request.prompt_text for request in batch],
                [request.response_text for request in batch],
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt",
            )
            encoded = {name: value.to(self.device) for name, value in encoded.items()}
            with torch.inference_mode():
                logits = self.model(**encoded).logits.to(torch.float32)
            if time.perf_counter() - started > self.config.timeout_seconds:
                raise TimeoutError(
                    f"reward-model scoring exceeded {self.config.timeout_seconds:g} seconds"
                )
            if logits.ndim != 2 or logits.shape[0] != len(batch):
                raise ValueError("reward model returned logits with an unexpected shape")
            if logits.shape[1] == 1:
                scores = logits[:, 0]
            else:
                index = self.config.positive_class_index
                if index >= logits.shape[1]:
                    raise ValueError(
                        f"positive_class_index={index} exceeds reward-model labels={logits.shape[1]}"
                    )
                scores = torch.softmax(logits, dim=-1)[:, index]
            for request, score in zip(batch, scores.tolist(), strict=True):
                if not math.isfinite(float(score)):
                    raise ValueError("reward model returned a non-finite score")
                outputs.append(
                    RewardResult(
                        trajectory_id=request.trajectory_id,
                        prompt_group_id=request.prompt_group_id,
                        sample_index=request.sample_index,
                        samples_per_prompt=request.samples_per_prompt,
                        provider=self.identity,
                        input_digest=request.input_digest,
                        raw_reward=float(score),
                        components=(
                            RewardComponent(
                                name="trained_reward_model",
                                value=float(score),
                                detail=f"{self.config.model_id}@{self.config.revision}",
                            ),
                        ),
                        status=RewardStatus.OK,
                        duration_ms=max((time.perf_counter() - started) * 1000.0, 0.0),
                        deterministic=True,
                    )
                )
        return outputs

    def to_device(self, device: str) -> None:
        """Place the reward role for its bounded inference phase."""
        self.model.to(device)
        self.device = device

    def release(self) -> None:
        """Move model state off the accelerator before trainer teardown."""
        self.model.to("cpu")
        self.device = "cpu"


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
