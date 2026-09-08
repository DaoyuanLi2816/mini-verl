"""Advantage and return estimators matching pinned verl v0.9.0 semantics."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AdvantageOutput:
    advantages: torch.Tensor
    returns: torch.Tensor
    sequence_advantages: torch.Tensor | None = None
    zero_variance_groups: int = 0


def _validate_batch(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
) -> None:
    if token_level_rewards.ndim != 2 or response_mask.shape != token_level_rewards.shape:
        raise ValueError(
            "token_level_rewards and response_mask must have identical [batch, time] shape"
        )
    if not torch.isfinite(token_level_rewards).all() or not torch.isfinite(response_mask).all():
        raise ValueError("rewards and masks must be finite")
    if bool((response_mask < 0).any()) or bool((response_mask > 1).any()):
        raise ValueError("response_mask must contain values in [0, 1]")
    if bool((response_mask.sum(dim=-1) == 0).any()):
        raise ValueError("every response must contain at least one model token")


def _validate_groups(groups: Sequence[str], batch_size: int) -> dict[str, list[int]]:
    if len(groups) != batch_size:
        raise ValueError("group identities must align with the response batch")
    counts = Counter(groups)
    if any(count < 2 for count in counts.values()):
        raise ValueError("group estimators require at least two complete samples per prompt")
    indices: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        if not group:
            raise ValueError("prompt group identity cannot be empty")
        indices[group].append(index)
    return dict(indices)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.where(mask.bool(), values, 0.0).sum() / (mask.sum() + 1e-8)


def _masked_whiten(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mean = _masked_mean(values, mask)
    centered = values - mean
    variance = _masked_mean(centered.square(), mask)
    count = mask.sum()
    if bool(count <= 1):
        raise ValueError("advantage whitening requires at least two selected tokens")
    variance = variance * count / (count - 1)
    return centered * torch.rsqrt(variance + 1e-8)


def grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    groups: Sequence[str],
    *,
    epsilon: float = 1e-6,
    normalize_by_std: bool = True,
) -> AdvantageOutput:
    """Compute upstream GRPO, or Dr.GRPO when ``normalize_by_std`` is false.

    verl v0.9.0 uses ``torch.std``'s sample standard deviation within each
    prompt group and adds epsilon after the standard deviation.
    """
    _validate_batch(token_level_rewards, response_mask)
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    grouped = _validate_groups(groups, token_level_rewards.shape[0])
    scores = token_level_rewards.sum(dim=-1).detach().clone()
    sequence = torch.empty_like(scores)
    zero_variance = 0
    for member_indices in grouped.values():
        index = torch.tensor(member_indices, device=scores.device)
        values = scores[index]
        centered = values - values.mean()
        std = values.std()
        if bool(std == 0):
            zero_variance += 1
        sequence[index] = centered / (std + epsilon) if normalize_by_std else centered
    advantages = sequence.unsqueeze(-1) * response_mask
    return AdvantageOutput(
        advantages=advantages,
        returns=advantages,
        sequence_advantages=sequence,
        zero_variance_groups=zero_variance,
    )


def rloo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    groups: Sequence[str],
) -> AdvantageOutput:
    """Compute verl v0.9.0 leave-one-out outcome advantages."""
    _validate_batch(token_level_rewards, response_mask)
    grouped = _validate_groups(groups, token_level_rewards.shape[0])
    scores = token_level_rewards.sum(dim=-1).detach().clone()
    sequence = torch.empty_like(scores)
    for member_indices in grouped.values():
        index = torch.tensor(member_indices, device=scores.device)
        values = scores[index]
        sequence[index] = values - (values.sum() - values) / (len(member_indices) - 1)
    advantages = sequence.unsqueeze(-1) * response_mask
    return AdvantageOutput(advantages, advantages, sequence)


def reinforce_plus_plus_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    gamma: float = 1.0,
) -> AdvantageOutput:
    """Compute discounted REINFORCE++ returns and globally whiten advantages."""
    _validate_batch(token_level_rewards, response_mask)
    if not 0 <= gamma <= 1:
        raise ValueError("gamma must be in [0, 1]")
    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return: torch.Tensor | int = 0
        for position in reversed(range(token_level_rewards.shape[1])):
            candidate = token_level_rewards[:, position] + gamma * running_return
            returns[:, position] = candidate * response_mask[:, position]
            running_return = candidate * response_mask[:, position] + running_return * (
                1 - response_mask[:, position]
            )
        advantages = _masked_whiten(returns, response_mask) * response_mask
    return AdvantageOutput(advantages=advantages, returns=returns)


def gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    gamma: float = 1.0,
    lam: float = 1.0,
) -> AdvantageOutput:
    """Compute verl v0.9.0 GAE, including observation-token carry semantics."""
    _validate_batch(token_level_rewards, response_mask)
    if values.shape != token_level_rewards.shape or not torch.isfinite(values).all():
        raise ValueError("values must be a finite tensor aligned with rewards")
    if not 0 <= gamma <= 1 or not 0 <= lam <= 1:
        raise ValueError("gamma and lam must be in [0, 1]")
    with torch.no_grad():
        next_values: torch.Tensor | int = 0
        last_gae: torch.Tensor | int = 0
        reversed_advantages: list[torch.Tensor] = []
        for position in reversed(range(token_level_rewards.shape[1])):
            delta = token_level_rewards[:, position] + gamma * next_values - values[:, position]
            candidate = delta + gamma * lam * last_gae
            next_values = (
                values[:, position] * response_mask[:, position]
                + (1 - response_mask[:, position]) * next_values
            )
            last_gae = (
                candidate * response_mask[:, position] + (1 - response_mask[:, position]) * last_gae
            )
            reversed_advantages.append(last_gae)
        raw_advantages = torch.stack(reversed_advantages[::-1], dim=1)
        returns = raw_advantages + values
        # Keep upstream's values outside the mask byte-for-byte; consumers mask
        # them in policy/value aggregation.
        advantages = _masked_whiten(raw_advantages, response_mask)
    return AdvantageOutput(advantages=advantages, returns=returns)
