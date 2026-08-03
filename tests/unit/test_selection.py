"""Budgeted teacher-position selection.

Invariants protected here: every selector returns only trainable positions,
selection is reproducible across processes (no salted ``hash``), budgets are
respected, and the reported query ratio is the honest ratio.
"""

from __future__ import annotations

import pytest

from miniverl.config.models import GateConfig, GateSignal, SelectionConfig, SelectorName
from miniverl.schemas.trajectory import (
    Span,
    SpanType,
    TerminationReason,
    Trajectory,
    Turn,
)
from miniverl.selection.selectors import (
    aggregate_selection_stats,
    derive_seed,
    select_positions,
)
from miniverl.trajectory.alignment import SEGMENT_KEY
from miniverl.trajectory.masks import build_masks

_LAYOUT = [
    (SpanType.SYSTEM, 0, 10),
    (SpanType.USER, 10, 20),
    (SpanType.ASSISTANT_TEXT, 20, 30),
    (SpanType.ASSISTANT_TOOL_CALL, 30, 40),
    (SpanType.TOOL_RESULT, 40, 50),
    (SpanType.ASSISTANT_FINAL, 50, 60),
]


def _trajectory(trajectory_id: str = "t0") -> Trajectory:
    spans = [
        Span(
            span_type=t,
            start=a,
            end=b,
            turn_id=0,
            text=f"{t.value}",
            metadata={SEGMENT_KEY: f"{t.value}:{a}"},
        )
        for t, a, b in _LAYOUT
    ]
    model, critical = build_masks(spans, 60)
    return Trajectory(
        trajectory_id=trajectory_id,
        task_id="task",
        environment="calculator",
        token_ids=list(range(60)),
        attention_mask=[1] * 60,
        model_generated_mask=model,
        critical_mask=critical,
        spans=spans,
        turns=[Turn(turn_id=0)],
        tokenizer_fingerprint="fp",
        model_id="m",
        termination_reason=TerminationReason.FINAL_ANSWER,
    )


def test_all_model_tokens_selects_exactly_the_trainable_positions():
    traj = _trajectory()
    result = select_positions(traj, SelectionConfig(selector=SelectorName.ALL_MODEL_TOKENS))
    expected = list(range(20, 40)) + list(range(50, 60))
    assert result.positions == expected
    assert result.stats.query_ratio == pytest.approx(1.0)
    assert result.stats.total_model_tokens == 30
    assert result.stats.by_span_type == {
        "assistant_text": 10,
        "assistant_tool_call": 10,
        "assistant_final": 10,
    }


def test_tool_and_final_selects_only_critical_positions():
    traj = _trajectory()
    result = select_positions(traj, SelectionConfig(selector=SelectorName.TOOL_AND_FINAL))
    assert result.positions == list(range(30, 40)) + list(range(50, 60))
    assert all(traj.critical_mask[p] for p in result.positions)
    assert result.stats.query_ratio == pytest.approx(20 / 30)
    assert result.stats.selected_critical_tokens == 20


@pytest.mark.parametrize("ratio", [0.1, 0.35, 0.5, 1.0])
def test_uniform_ratio_respects_the_budget(ratio):
    import math

    traj = _trajectory()
    result = select_positions(
        traj, SelectionConfig(selector=SelectorName.UNIFORM_RATIO, ratio=ratio)
    )
    assert len(result.positions) == min(30, math.ceil(ratio * 30))
    assert result.positions == sorted(set(result.positions))
    assert all(traj.model_generated_mask[p] for p in result.positions)


def test_uniform_budget_never_overshoots_the_declared_query_fraction() -> None:
    import math

    traj = _trajectory()
    result = select_positions(
        traj,
        SelectionConfig(selector=SelectorName.UNIFORM_BUDGET, ratio=0.49),
        run_seed=7,
    )
    assert len(result.positions) == math.floor(0.49 * 30)
    assert result.stats.query_ratio <= 0.49


def test_hybrid_keeps_every_critical_token_then_fills_the_budget():
    traj = _trajectory()
    result = select_positions(traj, SelectionConfig(selector=SelectorName.HYBRID, ratio=0.8))
    critical = set(range(30, 40)) | set(range(50, 60))
    assert critical <= set(result.positions)
    assert len(result.positions) == 24  # ceil(0.8 * 30)
    assert result.positions == sorted(result.positions)


def test_hybrid_never_drops_critical_tokens_even_below_their_count():
    traj = _trajectory()
    result = select_positions(traj, SelectionConfig(selector=SelectorName.HYBRID, ratio=0.1))
    critical = set(range(30, 40)) | set(range(50, 60))
    assert set(result.positions) == critical  # budget 3 < 20 critical -> all critical kept
    assert result.stats.query_ratio > 0.1


def test_selection_is_deterministic_across_calls_and_seed_dependent():
    traj = _trajectory()
    config = SelectionConfig(selector=SelectorName.UNIFORM_RATIO, ratio=0.4)
    a = select_positions(traj, config, run_seed=7).positions
    b = select_positions(traj, config, run_seed=7).positions
    c = select_positions(traj, config, run_seed=8).positions
    assert a == b
    assert a != c


def test_selection_depends_on_the_trajectory_id_not_on_object_identity():
    config = SelectionConfig(selector=SelectorName.UNIFORM_RATIO, ratio=0.4)
    a = select_positions(_trajectory("alpha"), config, run_seed=1).positions
    b = select_positions(_trajectory("alpha"), config, run_seed=1).positions
    c = select_positions(_trajectory("beta"), config, run_seed=1).positions
    assert a == b
    assert a != c


def test_derive_seed_is_stable_and_unsalted():
    """A salted hash would break cross-process reproducibility."""
    assert derive_seed(7, "abc") == derive_seed(7, "abc")
    assert derive_seed(7, "abc") != derive_seed(8, "abc")
    assert derive_seed(7, "abc") != derive_seed(7, "abd")
    # Value pinned so a change to the derivation is a visible, deliberate break.
    assert derive_seed(1234, "calc-train-0:v0:s3") == 755235821300022336


def test_max_positions_cap_is_applied():
    traj = _trajectory()
    result = select_positions(
        traj,
        SelectionConfig(selector=SelectorName.ALL_MODEL_TOKENS, max_positions_per_trajectory=7),
    )
    assert len(result.positions) == 7
    assert result.positions == list(range(20, 27))


def test_weights_follow_the_critical_flag():
    traj = _trajectory()
    result = select_positions(
        traj,
        SelectionConfig(
            selector=SelectorName.ALL_MODEL_TOKENS, critical_weight=3.0, other_weight=0.5
        ),
    )
    for position, weight in zip(result.positions, result.weights, strict=False):
        expected = 3.0 if traj.critical_mask[position] else 0.5
        assert weight == pytest.approx(expected)


def test_stats_to_dict_and_len():
    traj = _trajectory()
    result = select_positions(traj, SelectionConfig())
    payload = result.stats.to_dict()
    assert payload["selector"] == "all_model_tokens"
    assert payload["selected_model_tokens"] == len(result)
    assert 0.0 <= payload["query_ratio"] <= 1.0


def test_aggregate_selection_stats_sums_across_trajectories():
    stats = [select_positions(_trajectory(f"t{i}"), SelectionConfig()).stats for i in range(3)]
    aggregate = aggregate_selection_stats(stats)
    assert aggregate["trajectories"] == 3
    assert aggregate["total_model_tokens"] == 90
    assert aggregate["selected_model_tokens"] == 90
    assert aggregate["teacher_queried_position_ratio"] == pytest.approx(1.0)
    assert aggregate["selected_by_span_type"]["assistant_final"] == 30


def test_aggregate_of_nothing_is_zero_not_a_crash():
    aggregate = aggregate_selection_stats([])
    assert aggregate["trajectories"] == 0
    assert aggregate["teacher_queried_position_ratio"] == 0.0


def test_no_selector_ever_returns_a_context_token():
    traj = _trajectory()
    for selector in SelectorName:
        gate = (
            GateConfig(
                version="policy-span-v1",
                signal=GateSignal.POLICY_CRITICAL_SPAN,
                frozen_before_test=True,
            )
            if selector is SelectorName.VERIFIER_GATED
            else None
        )
        result = select_positions(
            traj,
            SelectionConfig(selector=selector, ratio=0.9, gate=gate),
            run_seed=3,
        )
        for position in result.positions:
            assert traj.model_generated_mask[position], (selector, position)
            assert position > 0


def test_verifier_gate_selects_only_policy_sensitive_critical_spans() -> None:
    sensitive = _trajectory("sensitive").model_copy(
        update={"metadata": {"policy_sensitive": True, "policy_category": "authorization"}}
    )
    benign = _trajectory("benign").model_copy(
        update={"metadata": {"policy_sensitive": False, "policy_category": "benign_completion"}}
    )
    config = SelectionConfig(
        selector=SelectorName.VERIFIER_GATED,
        gate=GateConfig(
            version="policy-span-v1",
            signal=GateSignal.POLICY_CRITICAL_SPAN,
            frozen_before_test=True,
        ),
    )
    selected = select_positions(sensitive, config)
    skipped = select_positions(benign, config)
    assert selected.positions == list(range(30, 40)) + list(range(50, 60))
    assert selected.stats.gate_decision == "qualified"
    assert skipped.positions == []
    assert skipped.stats.gate_decision == "not_qualified"
