"""Structured RecoveryBench trajectory and aggregate metrics."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from miniverl.schemas.trajectory import MODEL_GENERATED_SPAN_TYPES, Trajectory

__all__ = [
    "TrajectoryRecoveryMetrics",
    "aggregate_recovery_metrics",
    "trajectory_recovery_metrics",
]


@dataclass(frozen=True)
class TrajectoryRecoveryMetrics:
    had_tool_error: bool
    first_query_failed: bool
    injected_error_observed: bool
    natural_error_observed: bool
    recovered_after_tool_error: bool
    success_given_first_query_error: bool | None
    schema_call_after_error: bool
    repeated_same_failed_call: bool
    turns_after_first_error: int
    turns_to_recovery: int | None
    distinct_tool_errors: int
    valid_sql_execution_rate: float | None
    tokens_after_first_error: int
    query_calls: int
    valid_sql_executions: int
    intervention_kind: str
    strict_task_success: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _signature(name: str, arguments: dict[str, Any]) -> str:
    return json.dumps({"name": name, "arguments": arguments}, sort_keys=True, separators=(",", ":"))


def trajectory_recovery_metrics(trajectory: Trajectory) -> TrajectoryRecoveryMetrics:
    """Derive exact metrics only from typed calls, results, spans, and verification."""
    result_turns = [
        turn
        for turn in trajectory.turns
        if turn.tool_call is not None and turn.tool_result is not None
    ]
    failed = [turn for turn in result_turns if not turn.tool_result.ok]  # type: ignore[union-attr]
    failed_results = [turn.tool_result for turn in failed if turn.tool_result is not None]
    query_turns = [turn for turn in result_turns if turn.tool_call.name == "query"]  # type: ignore[union-attr]
    first_query_failed = bool(query_turns and not query_turns[0].tool_result.ok)  # type: ignore[union-attr]
    injected = any(result.intervention for result in failed_results)
    natural = any(
        not result.intervention and result.error_code == "SQL_EXECUTION_ERROR"
        for result in failed_results
    )
    solved = bool(trajectory.verification and trajectory.verification.solved)
    first_error_turn = failed[0].turn_id if failed else None
    later_turns = (
        [turn for turn in trajectory.turns if turn.turn_id > first_error_turn]
        if first_error_turn is not None
        else []
    )
    schema_after = any(
        turn.tool_call is not None and turn.tool_call.name == "schema" for turn in later_turns
    )
    failed_signatures = [
        _signature(turn.tool_call.name, turn.tool_call.arguments)  # type: ignore[union-attr]
        for turn in failed
    ]
    errors = {
        result.error_code or f"unstructured:{result.error or ''}" for result in failed_results
    }
    valid_queries = sum(bool(turn.tool_result.ok) for turn in query_turns)  # type: ignore[union-attr]
    tokens_after = 0
    if first_error_turn is not None:
        tokens_after = sum(
            span.length
            for span in trajectory.spans
            if span.turn_id > first_error_turn and span.span_type in MODEL_GENERATED_SPAN_TYPES
        )
    return TrajectoryRecoveryMetrics(
        had_tool_error=bool(failed),
        first_query_failed=first_query_failed,
        injected_error_observed=injected,
        natural_error_observed=natural,
        recovered_after_tool_error=bool(failed and solved),
        success_given_first_query_error=(solved if first_query_failed else None),
        schema_call_after_error=schema_after,
        repeated_same_failed_call=len(failed_signatures) != len(set(failed_signatures)),
        turns_after_first_error=len(later_turns),
        turns_to_recovery=(
            trajectory.turns[-1].turn_id - first_error_turn
            if first_error_turn is not None and solved and trajectory.turns
            else None
        ),
        distinct_tool_errors=len(errors),
        valid_sql_execution_rate=(valid_queries / len(query_turns) if query_turns else None),
        tokens_after_first_error=tokens_after,
        query_calls=len(query_turns),
        valid_sql_executions=valid_queries,
        intervention_kind=str(trajectory.metadata.get("intervention_kind", "none")),
        strict_task_success=solved,
    )


def _aggregate(records: list[TrajectoryRecoveryMetrics]) -> dict[str, Any]:
    tasks = len(records)
    errors = [record for record in records if record.had_tool_error]
    first_failures = [record for record in records if record.first_query_failed]
    recovered = [record for record in errors if record.recovered_after_tool_error]
    query_calls = sum(record.query_calls for record in records)
    valid_queries = sum(record.valid_sql_executions for record in records)
    turns = [
        record.turns_to_recovery for record in recovered if record.turns_to_recovery is not None
    ]
    return {
        "tasks": tasks,
        "strict_task_success_rate": (
            sum(record.strict_task_success for record in records) / tasks if tasks else None
        ),
        "recovery_after_error_rate": (
            sum(record.recovered_after_tool_error for record in errors) / len(errors)
            if errors
            else None
        ),
        "success_given_first_query_error": (
            sum(bool(record.success_given_first_query_error) for record in first_failures)
            / len(first_failures)
            if first_failures
            else None
        ),
        "valid_sql_execution_rate": valid_queries / query_calls if query_calls else None,
        "turns_to_recovery": sum(turns) / len(turns) if turns else None,
        "injected_error_tasks": sum(record.injected_error_observed for record in records),
        "natural_error_tasks": sum(record.natural_error_observed for record in records),
        "schema_call_after_error_rate": (
            sum(record.schema_call_after_error for record in errors) / len(errors)
            if errors
            else None
        ),
        "tokens_after_first_error_total": sum(
            record.tokens_after_first_error for record in records
        ),
    }


def aggregate_recovery_metrics(trajectories: list[Trajectory]) -> dict[str, Any]:
    records = [trajectory_recovery_metrics(trajectory) for trajectory in trajectories]
    payload = _aggregate(records)
    payload["controlled_intervention"] = _aggregate(
        [record for record in records if record.intervention_kind == "controlled_schema_refresh"]
    )
    payload["natural_error"] = _aggregate(
        [record for record in records if record.intervention_kind == "natural_sql_error"]
    )
    payload["no_intervention"] = _aggregate(
        [record for record in records if record.intervention_kind == "none"]
    )
    return payload
