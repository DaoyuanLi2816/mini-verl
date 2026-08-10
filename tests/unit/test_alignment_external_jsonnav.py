"""Retained tool utility is the real verifier rate over manifest-named tasks.

The superseded proxy (`1 - over_refusal`) is what forced preregistration
amendment 1, so these check the two properties that make the replacement
trustworthy: the number is an exact verifier success rate over real rollouts,
and the tasks scored are the ones the frozen manifest names.
"""

from __future__ import annotations

from typing import Any

import pytest

from miniverl.alignment_external.jsonnav_utility import (
    JSONNAV_TASK_PREFIX,
    JsonNavSettings,
    score_jsonnav_tasks,
    task_index_from_id,
)

# ------------------------------------------------------------------- task ids


@pytest.mark.parametrize(
    ("task_id", "expected"),
    [("jsonnav_utility-00000", 0), ("jsonnav_utility-00042", 42), ("jsonnav_utility-00255", 255)],
)
def test_a_suite_task_id_names_its_environment_task(task_id: str, expected: int) -> None:
    assert task_index_from_id(task_id) == expected


@pytest.mark.parametrize(
    "task_id",
    ["ifeval-00001", "jsonnav_utility-abc", "jsonnav_utility", "00042", "jsonnav-00042"],
)
def test_a_foreign_task_id_is_refused(task_id: str) -> None:
    """Scoring an id from another endpoint would silently measure the wrong tasks."""
    with pytest.raises(ValueError, match="not a JSONNav suite task id"):
        task_index_from_id(task_id)


# ------------------------------------------------------------------- settings


def test_the_frozen_settings_are_pinned_and_digested() -> None:
    settings = JsonNavSettings()

    assert settings.protocol_version == "v2"
    assert settings.verifier_version == "v2"
    assert settings.difficulty == "hard"
    assert settings.max_total_tokens == 768
    assert len(settings.digest()) == 64


def test_changing_any_setting_changes_the_digest() -> None:
    base = JsonNavSettings().digest()

    assert JsonNavSettings(difficulty="easy").digest() != base
    assert JsonNavSettings(protocol_version="v1").digest() != base
    assert JsonNavSettings(max_turns=8).digest() != base


# ------------------------------------------------------- scripted environment


class _Verification:
    def __init__(self, solved: bool) -> None:
        self.solved = solved
        self.failure_category = None if solved else "wrong_answer"


class _Trajectory:
    def __init__(self, solved: bool, index: int) -> None:
        self.verification = _Verification(solved)
        self.termination_reason = "final_answer"
        self.emitted_tool_calls = 2
        self.parsed_tool_calls = 2 if solved else 1
        self.generated_token_count = 30 + index
        self.token_ids = [1, 2, 3, index]


class _Environment:
    """Records which task indices were asked for."""

    def __init__(self, solved_indices: set[int]) -> None:
        self.solved_indices = solved_indices
        self.requested: list[int] = []
        self.settings_seen: list[tuple[str, str]] = []

    def generate_task(self, index: int, seed: int, *, difficulty: str, split: str) -> Any:
        self.requested.append(index)
        self.settings_seen.append((difficulty, split))
        return type("Task", (), {"task_id": f"env-{index}"})()


class _Runner:
    def __init__(self, environment: _Environment) -> None:
        self.environment = environment
        self.policy_versions: list[int] = []
        self.temperatures: list[float] = []

    def rollout(
        self, task: Any, *, policy_version: int, seed: int, temperature: float | None = None
    ) -> _Trajectory:
        index = int(str(task.task_id).split("-")[-1])
        self.policy_versions.append(policy_version)
        self.temperatures.append(temperature)
        return _Trajectory(index in self.environment.solved_indices, index)


def _score(task_ids: list[str], solved: set[int], **kwargs: Any) -> dict[str, Any]:
    environment = _Environment(solved)
    runner = _Runner(environment)
    result = score_jsonnav_tasks(
        backend=object(), task_ids=task_ids, environment=environment, runner=runner, **kwargs
    )
    result["_environment"] = environment
    result["_runner"] = runner
    return result


# --------------------------------------------------------------------- scoring


def test_the_metric_is_the_exact_verifier_success_rate() -> None:
    ids = [f"{JSONNAV_TASK_PREFIX}-{index:05d}" for index in range(8)]

    result = _score(ids, solved={0, 2, 4})

    assert result["tasks"] == 8
    assert result["solved"] == 3
    assert result["success_rate"] == 3 / 8
    assert result["metric"] == "exact_verifier_success_rate"


def test_only_the_manifest_task_ids_are_scored() -> None:
    """An ad-hoc range would not be provably disjoint from the final test."""
    ids = [f"{JSONNAV_TASK_PREFIX}-{index:05d}" for index in (7, 19, 200)]

    result = _score(ids, solved={19})

    assert result["_environment"].requested == [7, 19, 200]
    assert [record["suite_task_id"] for record in result["records"]] == ids


def test_generation_is_greedy_and_the_policy_version_is_pinned() -> None:
    result = _score([f"{JSONNAV_TASK_PREFIX}-{i:05d}" for i in range(3)], solved=set())

    assert set(result["_runner"].policy_versions) == {0}
    assert set(result["_runner"].temperatures) == {0.0}
    assert result["decoding"] == "greedy"
    assert result["policy_version"] == 0


def test_the_frozen_difficulty_and_split_reach_the_environment() -> None:
    result = _score([f"{JSONNAV_TASK_PREFIX}-00000"], solved=set())

    assert result["_environment"].settings_seen == [("hard", "eval")]


def test_every_task_gets_a_record_with_its_provenance() -> None:
    result = _score([f"{JSONNAV_TASK_PREFIX}-{i:05d}" for i in range(4)], solved={1})

    assert len(result["records"]) == 4
    record = result["records"][1]
    assert record["solved"] is True
    assert record["environment_task_id"] == "env-1"
    assert record["termination_reason"] == "final_answer"
    assert record["parse_valid_rate"] == 1.0
    assert len(record["trajectory_digest"]) == 64
    # A failed task keeps its failure category rather than reporting nothing.
    assert result["records"][0]["failure_category"] == "wrong_answer"


def test_an_empty_task_list_fails_closed() -> None:
    """Zero tasks would produce a 0/0 that could be read as a real zero."""
    with pytest.raises(ValueError, match="must name them"):
        score_jsonnav_tasks(backend=object(), task_ids=[])


def test_a_missing_verification_is_not_a_success() -> None:
    class _NoVerification(_Runner):
        def rollout(self, task: Any, **kwargs: Any) -> Any:
            trajectory = _Trajectory(False, 0)
            trajectory.verification = None
            return trajectory

    environment = _Environment(set())
    result = score_jsonnav_tasks(
        backend=object(),
        task_ids=[f"{JSONNAV_TASK_PREFIX}-00000"],
        environment=environment,
        runner=_NoVerification(environment),
    )

    assert result["solved"] == 0
    assert result["records"][0]["failure_category"] == "no_verification"


def test_reruns_are_identical_for_the_same_inputs() -> None:
    ids = [f"{JSONNAV_TASK_PREFIX}-{index:05d}" for index in range(6)]

    first = _score(ids, solved={1, 3})
    second = _score(ids, solved={1, 3})

    assert first["success_rate"] == second["success_rate"]
    assert [r["trajectory_digest"] for r in first["records"]] == [
        r["trajectory_digest"] for r in second["records"]
    ]
