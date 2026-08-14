"""Static OPD placement decisions cannot authorize an impossible runtime."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from miniverl.bridge.opd_capabilities import (
    assert_runtime_placement_legal,
    decide_placement,
)
from miniverl.errors import ConfigError


@given(
    requested=st.sampled_from(["auto", "dual_model_resident", "swap", "shared_backbone"]),
    student_quantization=st.sampled_from(["none", "nf4", "int8"]),
    teacher_quantization=st.sampled_from(["none", "nf4", "int8"]),
    resident_feasible=st.one_of(st.none(), st.booleans()),
    same_base=st.booleans(),
)
def test_executable_placement_is_statically_legal_at_runtime(
    requested: str,
    student_quantization: str,
    teacher_quantization: str,
    resident_feasible: bool | None,
    same_base: bool,
) -> None:
    try:
        decision = decide_placement(
            requested=requested,
            student_quantization=student_quantization,
            teacher_quantization=teacher_quantization,
            resident_feasible=resident_feasible,
            shared_backbone_feasible=same_base,
        )
    except ConfigError:
        return

    if not decision.executable_without_probe:
        assert decision.strategy in {"requires_probe", "infeasible"}
        return
    runtime_strategy = (
        "resident" if decision.strategy in {"dual_model_resident", "shared_backbone"} else "swap"
    )
    assert_runtime_placement_legal(
        strategy=runtime_strategy,
        student_quantization=student_quantization,
        teacher_quantization=teacher_quantization,
    )


@pytest.mark.parametrize("quantization", ["nf4", "int8"])
def test_unknown_quantized_auto_requires_probe_instead_of_illegal_swap(
    quantization: str,
) -> None:
    decision = decide_placement(
        requested="auto",
        student_quantization=quantization,
        teacher_quantization="none",
        resident_feasible=None,
        shared_backbone_feasible=False,
    )

    assert decision.strategy == "requires_probe"
    assert decision.placement_not_proven is True
    assert decision.executable_without_probe is False
    assert decision.swap_feasible is False
