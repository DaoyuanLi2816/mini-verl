"""Torch-free reward requests, results, and provider identities."""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "RewardComponent",
    "RewardProviderIdentity",
    "RewardRequest",
    "RewardResult",
    "RewardStatus",
]


def _digest(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RewardStatus(str, Enum):
    OK = "ok"
    ERROR = "error"
    NOT_APPLICABLE = "not_applicable"


class RewardComponent(_FrozenModel):
    name: str = Field(min_length=1)
    value: float
    detail: str | None = None


class RewardProviderIdentity(_FrozenModel):
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    config_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    package_name: str | None = None
    package_version: str | None = None
    deterministic: bool = True

    @property
    def digest(self) -> str:
        return _digest(self.model_dump(mode="json"))


class RewardRequest(_FrozenModel):
    schema_version: int = 1
    trajectory_id: str = Field(min_length=1)
    prompt_group_id: str = Field(min_length=1)
    sample_index: int = Field(ge=0)
    samples_per_prompt: int = Field(ge=1)
    row_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    response_text: str
    reward_model: Any
    ground_truth: Any
    data_source: str = Field(min_length=1)
    input_digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(
        cls,
        *,
        trajectory_id: str,
        prompt_group_id: str,
        sample_index: int,
        samples_per_prompt: int,
        row_digest: str,
        response_text: str,
        reward_model: Any,
        ground_truth: Any,
        data_source: str,
    ) -> RewardRequest:
        inputs = {
            "trajectory_id": trajectory_id,
            "prompt_group_id": prompt_group_id,
            "sample_index": sample_index,
            "samples_per_prompt": samples_per_prompt,
            "row_digest": row_digest,
            "response_text": response_text,
            "reward_model": reward_model,
            "ground_truth": ground_truth,
            "data_source": data_source,
        }
        return cls(**inputs, input_digest=_digest(inputs))

    @model_validator(mode="after")
    def _identity_is_consistent(self) -> RewardRequest:
        if self.sample_index >= self.samples_per_prompt:
            raise ValueError("sample_index must be smaller than samples_per_prompt")
        expected = _digest(self.model_dump(mode="json", exclude={"schema_version", "input_digest"}))
        if self.input_digest != expected:
            raise ValueError("reward input_digest does not bind the request inputs")
        return self


class RewardResult(_FrozenModel):
    schema_version: int = 1
    trajectory_id: str = Field(min_length=1)
    prompt_group_id: str = Field(min_length=1)
    sample_index: int = Field(ge=0)
    samples_per_prompt: int = Field(ge=1)
    provider: RewardProviderIdentity
    input_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    raw_reward: float | None
    components: tuple[RewardComponent, ...] = ()
    status: RewardStatus
    failure_category: str | None = None
    detail: str | None = None
    duration_ms: float = Field(ge=0.0)
    deterministic: bool

    @model_validator(mode="after")
    def _status_matches_value(self) -> RewardResult:
        if self.status is RewardStatus.OK and self.raw_reward is None:
            raise ValueError("successful reward results require raw_reward")
        if self.status is not RewardStatus.OK and self.raw_reward is not None:
            raise ValueError("non-success reward results cannot substitute a numeric reward")
        if self.raw_reward is not None and not (-float("inf") < self.raw_reward < float("inf")):
            raise ValueError("raw_reward must be finite")
        return self
