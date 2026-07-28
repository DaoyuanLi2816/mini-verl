"""Policy-competence metrics keep strict scoring primary and diagnostics explicit."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from miniverl.agent.loop import RolloutStats
from miniverl.evaluation.diagnostics import (
    lenient_answer_matches,
    lenient_diagnostic_success_rate,
)
from miniverl.schemas.trajectory import TerminationReason


def _trajectory(
    *,
    solved: bool,
    predicted: str,
    expected: str,
    valid_call: bool,
) -> Any:
    return SimpleNamespace(
        turns=[
            SimpleNamespace(tool_call=SimpleNamespace(valid=valid_call)),
            SimpleNamespace(tool_call=None),
        ],
        invalid_tool_calls=0 if valid_call else 1,
        generated_token_count=10,
        termination_reason=TerminationReason.FINAL_ANSWER,
        verification=SimpleNamespace(
            solved=solved,
            predicted=predicted,
            expected=expected,
            failure_category=None if solved else "wrong_answer",
        ),
    )


def test_lenient_answer_matching_only_repairs_presentation() -> None:
    assert lenient_answer_matches("<answer> 42 </answer>", "42")
    assert lenient_answer_matches("<answer> 1.0005 </answer>", "1")
    assert not lenient_answer_matches("<answer> 41 </answer>", "42")


def test_policy_metrics_distinguish_strict_and_lenient_success() -> None:
    strict = _trajectory(solved=True, predicted="42", expected="42", valid_call=True)
    presentation = _trajectory(
        solved=False,
        predicted="<answer>42</answer>",
        expected="42",
        valid_call=False,
    )
    trajectories = [strict, presentation]
    stats = RolloutStats()
    for trajectory in trajectories:
        stats.observe(trajectory)

    metrics = stats.to_dict()
    assert metrics["success_rate"] == pytest.approx(0.5)
    assert metrics["strict_task_success_rate"] == pytest.approx(0.5)
    assert lenient_diagnostic_success_rate(trajectories) == pytest.approx(1.0)
    assert metrics["tool_call_count"] == 2
    assert metrics["valid_tool_call_rate"] == pytest.approx(0.5)
    assert metrics["final_answer_format_validity_rate"] == pytest.approx(1.0)
