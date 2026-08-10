"""Retained tool utility, measured through the real agent runtime.

JSONNav is a multi-turn tool environment: the policy emits tool calls, the
environment answers, and an exact verifier decides at the end whether the final
answer is right. Scoring a single completion would measure a different thing
and call it retained tool utility -- which is exactly what the superseded
`1 - over_refusal` proxy did, and why preregistration amendment 1 exists.

So this drives `RolloutRunner` against `JsonNavEnvironment` and reports the
verifier's success rate over the task ids the frozen selection manifest names.
Two properties the study depends on:

* task ids come from the manifest, never from an ad-hoc range, so the tasks
  scored here are provably disjoint from the final test;
* generation is greedy and the policy version is pinned at zero, so a rerun
  reproduces the same outcomes.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

__all__ = [
    "JSONNAV_TASK_PREFIX",
    "JsonNavSettings",
    "score_jsonnav_tasks",
    "task_index_from_id",
]

JSONNAV_TASK_PREFIX = "jsonnav_utility"


class JsonNavSettings:
    """Frozen environment and rollout settings for the utility endpoint."""

    __slots__ = (
        "difficulty",
        "max_new_tokens_per_turn",
        "max_total_tokens",
        "max_turns",
        "prompt_style",
        "protocol_version",
        "split",
        "task_seed",
        "verifier_version",
    )

    def __init__(
        self,
        *,
        difficulty: str = "hard",
        split: str = "eval",
        protocol_version: str = "v2",
        prompt_style: str = "full",
        verifier_version: str = "v2",
        task_seed: int = 20260809,
        max_turns: int = 4,
        max_new_tokens_per_turn: int = 64,
        max_total_tokens: int = 768,
    ) -> None:
        self.difficulty = difficulty
        self.split = split
        self.protocol_version = protocol_version
        self.prompt_style = prompt_style
        self.verifier_version = verifier_version
        self.task_seed = int(task_seed)
        self.max_turns = int(max_turns)
        self.max_new_tokens_per_turn = int(max_new_tokens_per_turn)
        self.max_total_tokens = int(max_total_tokens)

    def as_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__slots__}

    def digest(self) -> str:
        payload = "|".join(f"{name}={getattr(self, name)}" for name in self.__slots__)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def task_index_from_id(task_id: str) -> int:
    """Environment task index encoded in a suite task id.

    The suite names JSONNav tasks `jsonnav_utility-00000`..., so the id both
    selects a task and records which one was scored.
    """
    prefix, _, suffix = str(task_id).rpartition("-")
    if prefix != JSONNAV_TASK_PREFIX or not suffix.isdigit():
        raise ValueError(
            f"{task_id!r} is not a JSONNav suite task id; expected {JSONNAV_TASK_PREFIX}-<digits>"
        )
    return int(suffix)


def score_jsonnav_tasks(
    *,
    backend: Any,
    task_ids: Sequence[str],
    settings: JsonNavSettings | None = None,
    environment: Any = None,
    runner: Any = None,
) -> dict[str, Any]:
    """Exact verifier success rate over the manifest's JSONNav task ids.

    ``environment`` and ``runner`` are injected only so the offline tests can
    drive the same code path with a scripted backend; the production call
    supplies neither.
    """
    if not task_ids:
        raise ValueError("no JSONNav task ids were supplied; the manifest must name them")

    config = settings or JsonNavSettings()

    if environment is None:
        from miniverl.environments.jsonnav import JsonNavEnvironment

        environment = JsonNavEnvironment(
            protocol_version=config.protocol_version,
            prompt_style=config.prompt_style,
        )
    if runner is None:
        from miniverl.agent.loop import RolloutRunner
        from miniverl.config.models import RolloutConfig

        runner = RolloutRunner(
            backend=backend,
            environment=environment,
            config=RolloutConfig(
                max_turns=config.max_turns,
                max_new_tokens_per_turn=config.max_new_tokens_per_turn,
                max_total_tokens=config.max_total_tokens,
                temperature=0.0,
            ),
        )

    records: list[dict[str, Any]] = []
    solved = 0
    for task_id in task_ids:
        index = task_index_from_id(task_id)
        task = environment.generate_task(
            index, config.task_seed, difficulty=config.difficulty, split=config.split
        )
        # policy_version is pinned: nothing is training during selection, and a
        # moving version would make the cache identity meaningless.
        trajectory = runner.rollout(task, policy_version=0, seed=config.task_seed, temperature=0.0)
        verification = trajectory.verification
        success = bool(verification.solved) if verification is not None else False
        solved += int(success)

        emitted = trajectory.emitted_tool_calls or 0
        parsed = trajectory.parsed_tool_calls or 0
        records.append(
            {
                "suite_task_id": task_id,
                "environment_task_id": getattr(task, "task_id", str(index)),
                "solved": success,
                "termination_reason": str(trajectory.termination_reason),
                "emitted_tool_calls": emitted,
                "parsed_tool_calls": parsed,
                "parse_valid_rate": (parsed / emitted) if emitted else None,
                "generated_tokens": trajectory.generated_token_count,
                "trajectory_digest": hashlib.sha256(
                    ",".join(str(t) for t in trajectory.token_ids).encode("utf-8")
                ).hexdigest(),
                "failure_category": (
                    verification.failure_category if verification is not None else "no_verification"
                ),
            }
        )

    return {
        "endpoint": JSONNAV_TASK_PREFIX,
        "metric": "exact_verifier_success_rate",
        "tasks": len(task_ids),
        "solved": solved,
        # A real rate over real rollouts. None only if there were no tasks,
        # which raises above rather than reaching here.
        "success_rate": solved / len(task_ids),
        "settings": config.as_dict(),
        "settings_digest": config.digest(),
        "decoding": "greedy",
        "policy_version": 0,
        "records": records,
    }
