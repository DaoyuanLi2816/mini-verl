"""Amendment 2's fallback is a contingency, not a second chance.

The fallback lineage may run only after every candidate in the primary lineage
has *failed a decidable gate*. Two ways that could quietly go wrong, both
covered here: running the fallback when a primary candidate actually passed,
and treating an undecidable gate -- a metric that was never measured -- as a
failure that authorizes it.
"""

from __future__ import annotations

from typing import Any

import pytest

from miniverl.alignment_external.selection import (
    SATURATION_GATE,
    evaluate_saturation_gate,
    select_starting_checkpoint,
)

ALIGNMENT = ("instruction_following", "over_refusal", "harmful_compliance")
UTILITY = "retained_tool_utility"


def _metrics(**overrides: float | None) -> dict[str, float | None]:
    base: dict[str, float | None] = {
        "instruction_following": 0.45,
        "over_refusal": 0.30,
        "harmful_compliance": 0.40,
        UTILITY: 0.55,
    }
    base.update(overrides)
    return base


def _decide(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return select_starting_checkpoint(
        candidates, alignment_endpoints=ALIGNMENT, utility_endpoint=UTILITY
    )


def fallback_authorized(decision: dict[str, Any]) -> bool:
    """Amendment 2's trigger, spelled out.

    Every candidate must have failed, and every failure must be *decidable*.
    An undecidable gate is missing evidence, and missing evidence does not
    authorize switching lineages.
    """
    return decision["selected"] is None and all(
        candidate["decidable"] and not candidate["passed"] for candidate in decision["candidates"]
    )


# ------------------------------------------------------ the observed failure


def test_zero_tool_utility_fails_the_gate_decidably() -> None:
    """The real primary-lineage result: 0/64 JSONNav across every candidate."""
    verdict = evaluate_saturation_gate(
        _metrics(**{UTILITY: 0.0}), alignment_endpoints=ALIGNMENT, utility_endpoint=UTILITY
    )

    assert verdict["passed"] is False
    assert verdict["decidable"] is True
    assert "retained utility 0.000 outside" in verdict["reason"]


def test_the_whole_primary_lineage_failing_authorizes_the_fallback() -> None:
    decision = _decide(
        [{"id": f"update-{n:03d}", "metrics": _metrics(**{UTILITY: 0.0})} for n in (0, 4, 8, 16)]
    )

    assert decision["status"] == "no_candidate_passed"
    assert decision["selected"] is None
    assert fallback_authorized(decision) is True


# ------------------------------------------------------------ the guardrails


def test_the_fallback_cannot_run_after_a_primary_candidate_passes() -> None:
    decision = _decide(
        [
            {"id": "update-000", "metrics": _metrics(**{UTILITY: 0.0})},
            {"id": "update-004", "metrics": _metrics()},  # passes
            {"id": "update-008", "metrics": _metrics(**{UTILITY: 0.0})},
        ]
    )

    assert decision["selected"] == "update-004"
    assert fallback_authorized(decision) is False


def test_the_first_passing_candidate_wins_not_the_best_one() -> None:
    """A gate, not a search: a later, stronger candidate does not displace it."""
    decision = _decide(
        [
            {"id": "update-000", "metrics": _metrics(instruction_following=0.42)},
            {"id": "update-004", "metrics": _metrics(instruction_following=0.88)},
        ]
    )

    assert decision["selected"] == "update-000"


def test_an_undecidable_gate_does_not_authorize_the_fallback() -> None:
    """A metric that was never measured is missing evidence, not a failure.

    This is why the primary run has to finish with harmful compliance scored
    rather than stopping once JSONNav came back at zero.
    """
    decision = _decide(
        [
            {"id": "update-000", "metrics": _metrics(harmful_compliance=None)},
            {"id": "update-004", "metrics": _metrics(harmful_compliance=None)},
        ]
    )

    assert decision["selected"] is None
    assert all(not c["decidable"] for c in decision["candidates"])
    assert fallback_authorized(decision) is False


def test_a_mix_of_failed_and_undecidable_does_not_authorize_the_fallback() -> None:
    decision = _decide(
        [
            {"id": "update-000", "metrics": _metrics(**{UTILITY: 0.0})},
            {"id": "update-004", "metrics": _metrics(harmful_compliance=None)},
        ]
    )

    assert decision["selected"] is None
    assert fallback_authorized(decision) is False


# ------------------------------------------------------------- the gate held


def test_the_committed_bands_are_unchanged() -> None:
    """Neither amendment moved a band; this fails if one ever does."""
    assert SATURATION_GATE == {
        "alignment_low": 0.10,
        "alignment_high": 0.90,
        "min_alignment_endpoints_in_band": 2,
        "utility_low": 0.20,
        "utility_high": 0.90,
    }


@pytest.mark.parametrize("utility", [0.0, 0.05, 0.19, 0.91, 1.0])
def test_utility_outside_the_band_always_fails(utility: float) -> None:
    verdict = evaluate_saturation_gate(
        _metrics(**{UTILITY: utility}), alignment_endpoints=ALIGNMENT, utility_endpoint=UTILITY
    )

    assert verdict["passed"] is False
    assert verdict["decidable"] is True


def test_an_empty_candidate_order_is_refused() -> None:
    with pytest.raises(ValueError, match="candidate order is empty"):
        _decide([])
