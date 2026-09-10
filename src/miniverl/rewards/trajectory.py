"""Shared reward ABI for training and validation without optimizer side effects."""

from __future__ import annotations

from typing import Any

from miniverl.errors import ConfigError
from miniverl.rewards.models import RewardRequest
from miniverl.schemas.trajectory import SpanType, Trajectory


def trajectory_reward_request(trajectory: Trajectory, tokenizer: Any) -> RewardRequest:
    metadata = trajectory.metadata
    if (
        trajectory.prompt_group_id is None
        or trajectory.sample_index is None
        or trajectory.samples_per_prompt is None
    ):
        raise ConfigError("reward request requires complete prompt-group identity")
    response = "".join(
        span.text
        for span in trajectory.spans
        if span.span_type in {SpanType.ASSISTANT_TEXT, SpanType.ASSISTANT_FINAL}
    )
    eos = getattr(getattr(tokenizer, "_tok", None), "eos_token", None)
    if metadata.get("raw_prompt") is not None and isinstance(eos, str) and eos:
        response = response.replace(eos, "")
    extra = metadata.get("extra_info")
    reward = metadata.get("reward_model")
    ground_truth = reward.get("ground_truth") if isinstance(reward, dict) else None
    if ground_truth is None and isinstance(extra, dict):
        ground_truth = extra.get("ground_truth")
    first = next(
        (index for index, flag in enumerate(trajectory.model_generated_mask) if flag),
        len(trajectory.token_ids),
    )
    return RewardRequest.create(
        trajectory_id=trajectory.trajectory_id,
        prompt_group_id=trajectory.prompt_group_id,
        sample_index=trajectory.sample_index,
        samples_per_prompt=trajectory.samples_per_prompt,
        row_digest=metadata["row_digest"],
        response_text=response,
        prompt_text=tokenizer.decode(trajectory.token_ids[:first]),
        reward_model=reward,
        ground_truth=ground_truth,
        data_source=metadata["data_source"],
        raw_prompt=metadata.get("raw_prompt"),
        extra_info=extra,
    )
