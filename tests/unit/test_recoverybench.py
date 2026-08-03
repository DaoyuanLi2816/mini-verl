"""RecoveryBench environment, provenance, and metric contracts."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest
import yaml

from miniverl.agent.loop import RolloutRunner, RolloutStats
from miniverl.config.models import RolloutConfig
from miniverl.environments.base import FailureCategory, ToolCall, make_splits
from miniverl.environments.registry import available_environments, make_environment
from miniverl.environments.sqlite_recovery import (
    RECOVERY_SCHEMA_TEMPLATES,
    SqliteRecoveryEnvironment,
    recovery_template_registry_digest,
)
from miniverl.evaluation.recovery import (
    aggregate_recovery_metrics,
    trajectory_recovery_metrics,
)
from miniverl.models.tokenizers import ToyTokenizer
from miniverl.trajectory.io import read_trajectories, write_trajectories
from miniverl.utils.runs import canonical_json

REPO_ROOT = Path(__file__).resolve().parents[2]


def _runner(environment: SqliteRecoveryEnvironment) -> RolloutRunner:
    from miniverl.models.toy import ToyBackend

    backend = ToyBackend(
        tokenizer=ToyTokenizer(),
        model_id="toy-recovery-oracle",
        hidden_size=16,
        num_layers=1,
        num_heads=2,
        intermediate_size=32,
        trainable=False,
    )
    return RolloutRunner(
        backend=backend,
        environment=environment,
        config=RolloutConfig(max_turns=8, max_new_tokens_per_turn=512, max_total_tokens=4096),
    )


def test_recovery_environment_is_registered_without_replacing_sqlite() -> None:
    assert available_environments() == [
        "calculator",
        "jsonnav",
        "sqlite",
        "sqlite_recovery",
        "tool_policy",
    ]
    assert make_environment("sqlite").name == "sqlite"
    recovery = make_environment("sqlite_recovery")
    try:
        assert recovery.name == "sqlite_recovery"
    finally:
        recovery.close()


def test_template_registry_has_twelve_structurally_disjoint_safe_templates() -> None:
    assert len(RECOVERY_SCHEMA_TEMPLATES) >= 12
    by_split = {
        split: {
            template.template_id
            for template in RECOVERY_SCHEMA_TEMPLATES
            if template.split == split
        }
        for split in ("train", "eval", "test")
    }
    assert all(len(ids) >= 4 for ids in by_split.values())
    assert by_split["train"].isdisjoint(by_split["eval"])
    assert by_split["train"].isdisjoint(by_split["test"])
    assert by_split["eval"].isdisjoint(by_split["test"])
    assert len({template.digest for template in RECOVERY_SCHEMA_TEMPLATES}) == len(
        RECOVERY_SCHEMA_TEMPLATES
    )
    assert {template.relationship_layout for template in RECOVERY_SCHEMA_TEMPLATES} == {
        "direct_fk",
        "association_table",
    }
    for template in RECOVERY_SCHEMA_TEMPLATES:
        assert template.version == 1
        for identifier in template.identifiers:
            assert re.fullmatch(r"[a-z][a-z0-9_]{0,47}", identifier), identifier
    assert re.fullmatch(r"[0-9a-f]{64}", recovery_template_registry_digest())


def test_preregistration_binds_the_frozen_registry_and_training_schedule() -> None:
    preregistration = yaml.safe_load(
        (REPO_ROOT / "benchmarks/preregistration/recoverybench-v1.yaml").read_text(encoding="utf-8")
    )
    generation = preregistration["task_generation"]
    assert generation["template_registry_digest"] == recovery_template_registry_digest()
    assert preregistration["seeds"]["model_and_training"] == [1234, 20260727, 20260801]
    assert len(preregistration["arms"]) == 6

    env = SqliteRecoveryEnvironment(protocol_version="v2")
    try:
        splits = make_splits(
            env,
            counts=generation["counts"],
            seed=generation["split_seed"],
            difficulty=generation["difficulty"],
        )
    finally:
        env.close()
    identity = [
        {
            "task_id": task.task_id,
            "split": task.split,
            "prompt": task.prompt,
            "template_id": task.metadata["template_id"],
            "template_digest": task.metadata["template_digest"],
            "database_seed": task.metadata["database_seed"],
            "task_kind": task.metadata["task_kind"],
            "intervention_kind": task.metadata["intervention_kind"],
        }
        for task in splits["train"]
    ]
    digest = hashlib.sha256(canonical_json(identity).encode("utf-8")).hexdigest()
    assert digest == generation["training_task_schedule_digest"]


@pytest.mark.parametrize("split", ["train", "eval", "test"])
def test_generated_tasks_bind_template_and_intervention_provenance(split: str) -> None:
    env = SqliteRecoveryEnvironment(protocol_version="v2")
    try:
        first = env.generate_task(0, 8128, difficulty="hard", split=split)
        again = env.generate_task(0, 8128, difficulty="hard", split=split)
        assert first == again
        assert first.metadata["template_id"] in {
            template.template_id
            for template in RECOVERY_SCHEMA_TEMPLATES
            if template.split == split
        }
        assert first.metadata["template_version"] == 1
        assert re.fullmatch(r"[0-9a-f]{64}", first.metadata["template_digest"])
        assert first.metadata["database_seed"] == first.metadata["db_seed"]
        assert first.metadata["task_kind"]
        assert first.metadata["intervention_kind"] == "controlled_schema_refresh"
        assert first.metadata["expected_tool_sequence_class"]
    finally:
        env.close()


def test_controlled_error_is_answer_free_deterministic_and_occurs_once() -> None:
    env = SqliteRecoveryEnvironment(protocol_version="v2")
    task = env.generate_task(0, 8128, difficulty="hard", split="train")
    try:
        env.reset(task)
        call = ToolCall("query", {"sql": task.metadata["reference_sql"]})
        first = env.step(call)
        assert not first.ok
        assert first.error_code == "SCHEMA_REFRESH_REQUIRED"
        assert first.retryable is True
        assert first.intervention is True
        assert first.failure_category is FailureCategory.TOOL_ERROR
        assert task.answer not in (first.error or "")
        assert task.answer not in first.result
        assert first.tool_result_metadata == {
            "intervention_kind": "controlled_schema_refresh",
            "occurrence": 1,
        }

        schema = env.step(ToolCall("schema", {}))
        assert schema.ok
        corrected = env.step(call)
        repeated = env.step(call)
        assert corrected.ok and repeated.ok
        assert env.verify(task.answer).solved
    finally:
        env.close()


def test_invalid_query_does_not_consume_the_controlled_intervention() -> None:
    env = SqliteRecoveryEnvironment(protocol_version="v2")
    task = env.generate_task(0, 8128, difficulty="hard", split="train")
    try:
        env.reset(task)
        natural = env.step(ToolCall("query", {"sql": "SELECT missing_column FROM nowhere"}))
        assert not natural.ok
        assert natural.error_code == "SQL_EXECUTION_ERROR"
        assert natural.retryable is True
        assert natural.intervention is False
        controlled = env.step(ToolCall("query", {"sql": task.metadata["reference_sql"]}))
        assert controlled.error_code == "SCHEMA_REFRESH_REQUIRED"
        assert controlled.intervention is True
    finally:
        env.close()


def test_natural_error_subset_has_no_injected_failure() -> None:
    env = SqliteRecoveryEnvironment(protocol_version="v2")
    task = env.generate_task(1, 8128, difficulty="hard", split="eval")
    try:
        assert task.metadata["intervention_kind"] == "natural_sql_error"
        env.reset(task)
        failed = env.step(ToolCall("query", {"sql": task.metadata["natural_error_sql"]}))
        assert not failed.ok
        assert failed.error_code == "SQL_EXECUTION_ERROR"
        assert failed.retryable is True
        assert failed.intervention is False
        corrected = env.step(ToolCall("query", {"sql": task.metadata["reference_sql"]}))
        assert corrected.ok
    finally:
        env.close()


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE anything",
        "WITH x AS (SELECT 1) DELETE FROM anything",
        "ATTACH DATABASE 'forbidden.db' AS forbidden",
        "PRAGMA database_list",
    ],
)
def test_recovery_environment_retains_sqlite_sandbox(sql: str) -> None:
    env = SqliteRecoveryEnvironment(protocol_version="v2")
    task = env.generate_task(2, 8128, difficulty="hard", split="test")
    try:
        env.reset(task)
        result = env.step(ToolCall("query", {"sql": sql}))
        assert not result.ok
        assert result.failure_category is FailureCategory.TOOL_ERROR
        assert result.intervention is False
        survivor = env.step(ToolCall("schema", {}))
        assert survivor.ok
    finally:
        env.close()


@pytest.mark.parametrize("index", [0, 1, 2, 3, 4, 5])
@pytest.mark.torch
def test_oracle_executes_and_records_real_recovery(index: int) -> None:
    env = SqliteRecoveryEnvironment(protocol_version="v2")
    task = env.generate_task(index, 8128, difficulty="hard", split="train")
    try:
        trajectory = _runner(env).oracle_rollout(task)
    finally:
        env.close()
    assert trajectory.verification is not None and trajectory.verification.solved
    if task.metadata["intervention_kind"] == "controlled_schema_refresh":
        failures = [
            turn for turn in trajectory.turns if turn.tool_result and not turn.tool_result.ok
        ]
        assert len(failures) == 1
        assert failures[0].tool_result is not None
        assert failures[0].tool_result.error_code == "SCHEMA_REFRESH_REQUIRED"
        assert failures[0].tool_result.intervention is True
        failed_index = trajectory.turns.index(failures[0])
        assert any(
            turn.tool_call and turn.tool_call.name == "schema"
            for turn in trajectory.turns[failed_index + 1 :]
        )
    elif task.metadata["intervention_kind"] == "natural_sql_error":
        assert any(
            turn.tool_result
            and turn.tool_result.error_code == "SQL_EXECUTION_ERROR"
            and not turn.tool_result.intervention
            for turn in trajectory.turns
        )


@pytest.mark.torch
def test_structured_recovery_provenance_round_trips_and_reads_v1(tmp_path: Path) -> None:
    env = SqliteRecoveryEnvironment(protocol_version="v2")
    task = env.generate_task(0, 8128, difficulty="hard", split="train")
    try:
        trajectory = _runner(env).oracle_rollout(task)
    finally:
        env.close()
    assert trajectory.schema_version == 2
    path = tmp_path / "recovery.jsonl"
    write_trajectories(path, [trajectory])
    loaded = read_trajectories(path)[0]
    failed = next(
        turn.tool_result for turn in loaded.turns if turn.tool_result and not turn.tool_result.ok
    )
    assert failed.error_code == "SCHEMA_REFRESH_REQUIRED"
    assert failed.retryable is True
    assert failed.intervention is True
    assert failed.tool_result_metadata["occurrence"] == 1

    payload = trajectory.model_dump(mode="json")
    payload["schema_version"] = 1
    for turn in payload["turns"]:
        if turn["tool_result"] is not None:
            for key in ("error_code", "retryable", "intervention", "tool_result_metadata"):
                turn["tool_result"].pop(key, None)
    legacy = tmp_path / "legacy-v1.jsonl"
    legacy.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    legacy_loaded = read_trajectories(legacy)[0]
    assert legacy_loaded.schema_version == 1
    assert all(
        turn.tool_result is None or turn.tool_result.error_code is None
        for turn in legacy_loaded.turns
    )


@pytest.mark.torch
def test_recovery_metrics_are_derived_from_structured_turns() -> None:
    env = SqliteRecoveryEnvironment(protocol_version="v2")
    task = env.generate_task(0, 8128, difficulty="hard", split="train")
    try:
        trajectory = _runner(env).oracle_rollout(task)
    finally:
        env.close()
    metrics = trajectory_recovery_metrics(trajectory)
    assert metrics.had_tool_error
    assert metrics.first_query_failed
    assert metrics.injected_error_observed
    assert not metrics.natural_error_observed
    assert metrics.recovered_after_tool_error
    assert metrics.success_given_first_query_error is True
    assert metrics.schema_call_after_error
    assert not metrics.repeated_same_failed_call
    assert metrics.turns_after_first_error == 3
    assert metrics.turns_to_recovery == 3
    assert metrics.distinct_tool_errors == 1
    assert metrics.valid_sql_execution_rate == pytest.approx(0.5)
    assert metrics.tokens_after_first_error > 0

    aggregate = aggregate_recovery_metrics([trajectory])
    assert aggregate["recovery_after_error_rate"] == pytest.approx(1.0)
    assert aggregate["success_given_first_query_error"] == pytest.approx(1.0)
    assert aggregate["controlled_intervention"]["tasks"] == 1
    assert aggregate["natural_error"]["tasks"] == 0

    stats = RolloutStats()
    stats.observe(trajectory)
    evaluation = stats.to_dict()
    assert evaluation["recovery_after_error_rate"] == pytest.approx(1.0)
    assert evaluation["success_given_first_query_error"] == pytest.approx(1.0)
    assert evaluation["recovery_subsets"]["controlled_intervention"]["tasks"] == 1
