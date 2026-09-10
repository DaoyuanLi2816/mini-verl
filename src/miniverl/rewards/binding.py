"""Explicit, checksum-approved local implementation of the verl reward ABI."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

from miniverl.errors import ConfigError
from miniverl.rewards.models import (
    RewardProviderIdentity,
    RewardRequest,
    RewardResult,
    RewardStatus,
)


class BoundVerlReward:
    """Load code only from a CLI-supplied path whose exact bytes were approved.

    This is trusted code, not a sandbox. Neither upstream YAML nor an imported
    report can authorize execution. Recheck on every process start/resume.
    """

    def __init__(self, binding: str, *, approved_sha256: str) -> None:
        filename, separator, function = binding.rpartition(":")
        if not separator or not function.isidentifier():
            raise ConfigError("reward.function must be a local file.py:function_name")
        path = Path(filename).resolve()
        if path.suffix != ".py" or not path.is_file():
            raise ConfigError("reward.function must name an existing Python file")
        source = path.read_bytes()
        digest = hashlib.sha256(source).hexdigest()
        if digest != approved_sha256:
            raise ConfigError("reward code checksum does not match --trust-reward-code")
        namespace: dict[str, Any] = {"__name__": "miniverl_bound_reward", "__file__": str(path)}
        exec(compile(source, str(path), "exec"), namespace)
        scorer = namespace.get(function)
        if not callable(scorer):
            raise ConfigError("bound reward function is not callable")
        self.scorer = scorer
        self.identity = RewardProviderIdentity(
            name=function,
            version="verl-python-abi-v1",
            config_digest=digest,
            package_name="explicit-local-binding",
            deterministic=False,
        )

    def score(self, request: RewardRequest) -> RewardResult:
        value = self.scorer(
            data_source=request.data_source,
            solution_str=request.response_text,
            ground_truth=request.ground_truth,
            extra_info=request.extra_info,
        )
        if isinstance(value, dict):
            value = value.get("score")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ConfigError("bound reward must return a finite scalar or {'score': scalar}")
        return RewardResult(
            trajectory_id=request.trajectory_id,
            prompt_group_id=request.prompt_group_id,
            sample_index=request.sample_index,
            samples_per_prompt=request.samples_per_prompt,
            provider=self.identity,
            input_digest=request.input_digest,
            raw_reward=float(value),
            status=RewardStatus.OK,
            duration_ms=0.0,
            deterministic=self.identity.deterministic,
        )
