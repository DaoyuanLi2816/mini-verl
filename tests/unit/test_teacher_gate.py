"""Teacher qualification stays eval-only and applies every preregistered gate."""

from __future__ import annotations

import pytest

from miniverl.errors import ConfigError
from miniverl.evaluation.teacher_gate import apply_teacher_gate, validate_gate_split

GATE = {
    "strict_task_success_rate": 0.80,
    "recovery_after_error_rate": 0.75,
    "parse_valid_tool_call_rate": 0.95,
    "tool_execution_success_rate": 0.70,
}


def test_teacher_gate_requires_every_metric_and_retains_checks() -> None:
    result = apply_teacher_gate(
        {
            "strict_task_success_rate": 0.90,
            "recovery_after_error_rate": 0.80,
            "parse_valid_tool_call_rate": 0.96,
            "tool_execution_success_rate": 0.69,
        },
        GATE,
    )
    assert result["passed"] is False
    assert set(result["checks"]) == set(GATE)
    assert result["checks"]["tool_execution_success_rate"]["passed"] is False


def test_teacher_gate_missing_metric_fails_closed() -> None:
    result = apply_teacher_gate({"strict_task_success_rate": 1.0}, GATE)
    assert result["passed"] is False
    assert result["checks"]["recovery_after_error_rate"]["actual"] is None


def test_teacher_qualification_refuses_final_test_split() -> None:
    assert validate_gate_split("eval") == "eval"
    with pytest.raises(ConfigError, match="eval split only"):
        validate_gate_split("test")
