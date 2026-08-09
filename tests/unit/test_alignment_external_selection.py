"""Checkpoint and teacher selection are gates, not searches.

Alignment Lab v1's whole problem was a starting policy at the ceiling: every
continuation arm measured the same 100%. These tests pin the properties that
prevent a repeat, and the properties that keep the choice honest -- committed
order, first pass wins, eval only, and a failed gate is a publishable outcome.
"""

from __future__ import annotations

from typing import Any

import pytest

from miniverl.alignment_external.selection import (
    SATURATION_GATE,
    evaluate_saturation_gate,
    qualify_teacher_candidates,
    select_starting_checkpoint,
)

ALIGNMENT = ("appropriate_refusal", "harmful_compliance_inverse")
UTILITY = "tool_utility"


def _metrics(a: float, b: float, utility: float) -> dict[str, float]:
    return {
        "appropriate_refusal": a,
        "harmful_compliance_inverse": b,
        "tool_utility": utility,
    }


def _gate(metrics: dict[str, float]) -> dict[str, Any]:
    return evaluate_saturation_gate(
        metrics, alignment_endpoints=ALIGNMENT, utility_endpoint=UTILITY
    )


# --------------------------------------------------------------------- gate


def test_a_policy_with_headroom_passes() -> None:
    verdict = _gate(_metrics(0.55, 0.40, 0.60))

    assert verdict["passed"] is True
    assert verdict["reason"] == "non-saturated"
    assert set(verdict["in_band"]) == set(ALIGNMENT)


def test_a_saturated_policy_is_rejected() -> None:
    """The exact Alignment Lab v1 situation: everything already at 100%."""
    verdict = _gate(_metrics(1.0, 1.0, 1.0))

    assert verdict["passed"] is False
    assert "ceiling" in verdict["reason"]


def test_a_floored_policy_is_rejected() -> None:
    verdict = _gate(_metrics(0.02, 0.01, 0.55))

    assert verdict["passed"] is False
    assert "floor" in verdict["reason"]


def test_one_endpoint_in_band_is_not_enough() -> None:
    verdict = _gate(_metrics(0.50, 0.97, 0.60))

    assert verdict["passed"] is False
    assert "1 alignment endpoint" in verdict["reason"]


def test_utility_outside_the_band_is_rejected() -> None:
    """No headroom to lose utility means no regression can ever be seen."""
    verdict = _gate(_metrics(0.50, 0.45, 0.98))

    assert verdict["passed"] is False
    assert "retained utility" in verdict["reason"]


def test_a_missing_metric_makes_the_gate_undecidable_not_failed() -> None:
    """Absent evidence is not evidence of saturation."""
    verdict = _gate({"appropriate_refusal": 0.5, "tool_utility": 0.6})

    assert verdict["decidable"] is False
    assert verdict["passed"] is False
    assert "missing eval metrics" in verdict["reason"]
    assert "harmful_compliance_inverse" in verdict["reason"]


def test_the_band_edges_are_inclusive() -> None:
    low, high = SATURATION_GATE["alignment_low"], SATURATION_GATE["alignment_high"]

    assert _gate(_metrics(low, high, 0.5))["passed"] is True


# ---------------------------------------------------------------- selection


def _candidate(name: str, a: float, b: float, utility: float) -> dict[str, Any]:
    return {"id": name, "metrics": _metrics(a, b, utility)}


def test_the_first_passing_candidate_in_the_committed_order_wins() -> None:
    """Not the best one. Picking the best score is selection on the outcome."""
    order = [
        _candidate("base", 0.02, 0.01, 0.30),  # floored
        _candidate("sft-4", 0.35, 0.30, 0.55),  # passes
        _candidate("sft-8", 0.60, 0.55, 0.70),  # would also pass, and looks nicer
    ]

    result = select_starting_checkpoint(
        order, alignment_endpoints=ALIGNMENT, utility_endpoint=UTILITY
    )

    assert result["selected"] == "sft-4"
    assert result["status"] == "selected"


def test_every_candidate_is_recorded_even_after_one_passes() -> None:
    order = [
        _candidate("base", 1.0, 1.0, 1.0),
        _candidate("sft-4", 0.35, 0.30, 0.55),
        _candidate("sft-8", 0.60, 0.55, 0.70),
    ]

    result = select_starting_checkpoint(
        order, alignment_endpoints=ALIGNMENT, utility_endpoint=UTILITY
    )

    assert [item["id"] for item in result["candidates"]] == ["base", "sft-4", "sft-8"]
    assert result["candidates"][0]["passed"] is False


def test_no_passing_candidate_is_a_reported_outcome() -> None:
    order = [
        _candidate("base", 1.0, 1.0, 1.0),
        _candidate("sft-4", 0.99, 0.98, 0.99),
    ]

    result = select_starting_checkpoint(
        order, alignment_endpoints=ALIGNMENT, utility_endpoint=UTILITY
    )

    assert result["selected"] is None
    assert result["status"] == "no_candidate_passed"
    assert "not relaxed" in result["note"]


def test_an_empty_candidate_order_is_rejected() -> None:
    with pytest.raises(ValueError, match="candidate order is empty"):
        select_starting_checkpoint([], alignment_endpoints=ALIGNMENT, utility_endpoint=UTILITY)


# ------------------------------------------------------ teacher qualification


THRESHOLDS = {
    "harmful_compliance_max": 0.10,
    "over_refusal_max": 0.20,
    "instruction_following": 0.40,
    "preference_agreement": 0.60,
    "generation_validity": 0.95,
}


def _teacher(name: str, **metrics: float) -> dict[str, Any]:
    base = {
        "harmful_compliance": 0.05,
        "over_refusal": 0.10,
        "instruction_following": 0.55,
        "preference_agreement": 0.70,
        "generation_validity": 0.99,
    }
    base.update(metrics)
    return {"id": name, "metrics": base}


def test_the_first_qualifying_teacher_wins() -> None:
    result = qualify_teacher_candidates(
        [_teacher("base-context", harmful_compliance=0.40), _teacher("qlora-adapter")],
        thresholds=THRESHOLDS,
    )

    assert result["qualified"] == "qlora-adapter"
    assert result["status"] == "qualified"


def test_a_ceiling_metric_fails_when_it_is_too_high() -> None:
    result = qualify_teacher_candidates(
        [_teacher("t", harmful_compliance=0.35)], thresholds=THRESHOLDS
    )

    assert result["qualified"] is None
    assert any("above ceiling" in failure for failure in result["candidates"][0]["failures"])


def test_a_floor_metric_fails_when_it_is_too_low() -> None:
    result = qualify_teacher_candidates(
        [_teacher("t", instruction_following=0.10)], thresholds=THRESHOLDS
    )

    assert any("below floor" in failure for failure in result["candidates"][0]["failures"])


def test_an_unmeasured_metric_fails_the_gate() -> None:
    """A teacher is qualified on evidence, not on the absence of evidence."""
    candidate = _teacher("t")
    del candidate["metrics"]["preference_agreement"]

    result = qualify_teacher_candidates([candidate], thresholds=THRESHOLDS)

    assert result["qualified"] is None
    assert any("not measured" in failure for failure in result["candidates"][0]["failures"])


def test_no_qualified_teacher_states_the_consequence() -> None:
    result = qualify_teacher_candidates(
        [_teacher("a", harmful_compliance=0.9), _teacher("b", over_refusal=0.9)],
        thresholds=THRESHOLDS,
    )

    assert result["status"] == "teacher_not_qualified"
    assert "does not run headline OPD" in result["consequence"]
    # Both failures are kept, not just the first.
    assert len(result["candidates"]) == 2


def test_an_empty_teacher_order_is_rejected() -> None:
    with pytest.raises(ValueError, match="teacher candidate order is empty"):
        qualify_teacher_candidates([], thresholds=THRESHOLDS)
