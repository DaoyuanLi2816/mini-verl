"""Pinned v5 epoch ordering and synchronous group selection primitives."""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any, TypeVar

from miniverl.bridge.direct_v5 import PROFILE, SamplingSemantics
from miniverl.errors import ConfigError

T = TypeVar("T")


class RolloutGroupFailure(RuntimeError):
    """A terminal rollout group has no materializable trajectories; retryable if bound."""

    prompt_group_ids: tuple[str, ...] = ()


def run_sampling(config: Any) -> SamplingSemantics | None:
    identity = getattr(getattr(config, "run", None), "profile_identity", None) or {}
    if identity.get("profile_name") != PROFILE:
        return None
    return SamplingSemantics.model_validate(identity.get("v5_sampling", {}))


def epoch_samples(
    samples: list[T],
    *,
    minibatch: int,
    epochs: int,
    shuffle: bool,
    seed: int,
) -> Iterator[list[T]]:
    """Use the same torch DataLoader iterator as upstream's TensorDict worker.

    A new generator is seeded per role/update call, and advances across epochs.
    This includes DataLoader's base-seed consumption; randperm alone differs.
    Logical trajectories are shuffled, not packed microbatches or prompt IDs.
    """
    import torch
    from torch.utils.data import DataLoader

    if not samples or len(samples) % minibatch:
        raise ConfigError("upstream minibatch iteration requires an evenly divisible rollout batch")
    loader: DataLoader[Any] = DataLoader(
        torch.arange(len(samples)),  # type: ignore[arg-type]
        batch_size=minibatch,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=lambda items: items,
    )
    for _ in range(epochs):
        yield [samples[int(index)] for batch in loader for index in batch]


def group_decision(trajectories: list[Any], metric: str | None) -> str:
    """Filter a complete group by the raw reward metric, before normalization."""
    if not trajectories:
        return "failed_empty_group"
    expected = trajectories[0].samples_per_prompt
    if len(trajectories) != expected or sorted(t.sample_index for t in trajectories) != list(
        range(expected)
    ):
        raise ConfigError("group filtering cannot consume incomplete rollout groups")
    if metric is None:
        return "retained"
    values = []
    for trajectory in trajectories:
        reward = trajectory.metadata.get("task_reward", {})
        components = {item["name"]: item["value"] for item in reward.get("components", [])}
        value = reward.get("raw_reward") if metric == "score" else components.get(metric)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ConfigError(f"group filtering requires finite reward metric {metric!r}")
        values.append(float(value))
    # The pinned replay buffer keeps singleton groups; np.std(values)==0 for
    # every exactly equal finite group. No reward normalization enters here.
    import numpy as np

    return (
        "filtered_zero_variance" if len(values) > 1 and float(np.std(values)) == 0.0 else "retained"
    )


def collect_selected_groups(host: Any, settings: SamplingSemantics) -> tuple[list[Any], Any]:
    """Drain synchronous dispatch waves, evict groups and refill before any update.

    DAPO refill credit is twice evicted groups plus fully failed groups, matching
    the pinned V1 buffer. Equal-age surplus has a recorded local tie-break:
    dispatch order. The upstream set iteration does not define that tie-break.
    """
    from miniverl.agent.loop import RolloutStats
    from miniverl.rewards import RewardResult

    target = host.config.train.rollouts_per_cycle
    policy = host.parameter_version
    retained: list[list[Any]] = []
    attempts = filtered = failed = credit = 0
    dispatch = target
    all_stats = RolloutStats()
    execution: dict[str, Any] = {
        "physical_batch_sizes": [],
        "physical_generation_batches": 0,
        "oom_downshifts": 0,
    }
    while True:
        evicted_now = failed_now = 0
        for _ in range(dispatch):
            if (
                settings.max_attempted_groups is not None
                and attempts >= settings.max_attempted_groups
            ):
                raise ConfigError(
                    "local sampling.max_attempted_groups guard exhausted before a full retained batch; "
                    "no partial optimizer batch was committed"
                )
            tasks = host._next_tasks(1)
            group_cursor = host._rollout_group_cursor
            attempts += 1
            if host.parameter_version != policy:
                raise ConfigError("refill cannot cross a policy update")
            try:
                rows, _ = host._collect(tasks, oracle=False)
            except RolloutGroupFailure as exc:
                if not settings.refill_failed_groups:
                    raise
                # _collect rolled back the failed group's input cursor. Consume
                # its identity once; the replacement is the next dataset group.
                host.task_cursor += 1
                host._committed_task_cursor = host.task_cursor
                host._rollout_group_cursor += 1
                host._prompt_train_iterator = None
                failed += 1
                failed_now += 1
                decision = "replaced_failed_group"
                rows = []
                group_id = exc.prompt_group_ids[0] if exc.prompt_group_ids else None
            else:
                physical = host._last_rollout_execution or {}
                execution["physical_batch_sizes"].extend(physical.get("physical_batch_sizes", []))
                execution["physical_generation_batches"] += physical.get(
                    "physical_generation_batches", 0
                )
                execution["oom_downshifts"] += physical.get("oom_downshifts", 0)
                host._score_task_rewards(rows, defer_advantages=True)
                decision = group_decision(rows, settings.filter_metric)
                group_id = rows[0].prompt_group_id
                for row in rows:
                    all_stats.observe(row)
                if decision == "filtered_zero_variance":
                    filtered += 1
                    evicted_now += 1
                else:
                    retained.append(rows)
            host.metrics_log.write(
                {
                    "phase": "rl_group_selection",
                    "cycle": host.cycle,
                    "policy_version": policy,
                    "group_cursor": group_cursor,
                    "prompt_group_id": group_id,
                    "trajectory_ids": [t.trajectory_id for t in rows],
                    "decision": decision,
                    "metric": settings.filter_metric,
                    "attempted_groups": attempts,
                    "retained_groups": len(retained),
                    "filtered_groups": filtered,
                    "failed_groups": failed,
                }
            )
        if len(retained) >= target:
            break
        credit += 2 * evicted_now + failed_now if settings.filter_metric else failed_now
        dispatch = min(credit, settings.max_inflight_gen_batches * target)
        credit -= dispatch
        if dispatch < 1:
            raise ConfigError("group refill made no progress toward the required logical batch")
    surplus = retained[target:]
    selected = [row for group in retained[:target] for row in group]
    for group in surplus:
        host.metrics_log.write(
            {
                "phase": "rl_group_selection",
                "cycle": host.cycle,
                "policy_version": policy,
                "decision": "discarded_surplus_group",
                "prompt_group_id": group[0].prompt_group_id,
                "trajectory_ids": [row.trajectory_id for row in group],
            }
        )
    host.metrics_log.write(
        {
            "phase": "rl_sampling",
            "cycle": host.cycle,
            "policy_version": policy,
            "attempted_groups": attempts,
            "retained_groups": target,
            "filtered_groups": filtered,
            "failed_groups": failed,
            "discarded_surplus_groups": len(surplus),
            "tie_break": "dispatch_order",
            "retry_budget": settings.max_attempted_groups,
            "selected_trajectory_ids": [row.trajectory_id for row in selected],
        }
    )
    # Estimators and global whitening see the final retained batch only. This
    # matters for PPO/REINFORCE++; group-wise normalization alone would hide it.
    if host.config.algorithm.name.value == "ppo":
        host._score_critic_values(selected)
    groups = {
        group[0].prompt_group_id: [
            (row, RewardResult.model_validate(row.metadata["task_reward"])) for row in group
        ]
        for group in retained[:target]
    }
    host._attach_rl_advantages(groups)
    host._last_rollout_execution = execution
    return selected, all_stats
