"""Adversarial tests for token provenance and causal alignment.

These are the tests that make "tool outputs are context, not labels" and "the
teacher scores position j-1 for target j" *checked properties* rather than
comments.  Each one is written to fail loudly if the corresponding invariant is
ever weakened.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from miniverl.errors import AlignmentError, TokenizerMismatchError, TrajectoryError
from miniverl.schemas.trajectory import (
    CRITICAL_SPAN_TYPES,
    MODEL_GENERATED_SPAN_TYPES,
    Span,
    SpanType,
    TerminationReason,
    Trajectory,
    Turn,
)
from miniverl.trajectory.alignment import SEGMENT_KEY, build_alignment_map, identity_alignment
from miniverl.trajectory.masks import (
    build_masks,
    critical_target_positions,
    model_target_positions,
    positions_by_span_type,
    prediction_positions,
    validate_target_positions,
)

# Layout used throughout: 0-3 system, 4-7 user, 8-11 tool call, 12-15 tool result,
# 16-19 final answer.
_LAYOUT: list[tuple[SpanType, int, int]] = [
    (SpanType.SYSTEM, 0, 4),
    (SpanType.USER, 4, 8),
    (SpanType.ASSISTANT_TOOL_CALL, 8, 12),
    (SpanType.TOOL_RESULT, 12, 16),
    (SpanType.ASSISTANT_FINAL, 16, 20),
]


def _spans(layout=_LAYOUT) -> list[Span]:
    return [
        Span(
            span_type=span_type,
            start=start,
            end=end,
            turn_id=0,
            text=f"{span_type.value}[{start}:{end}]",
            metadata={SEGMENT_KEY: f"{span_type.value}:{start}"},
        )
        for span_type, start, end in layout
    ]


def _trajectory(**overrides) -> Trajectory:
    spans = overrides.pop("spans", _spans())
    length = spans[-1].end
    model, critical = build_masks(spans, length)
    payload: dict = {
        "trajectory_id": "t0",
        "task_id": "task0",
        "environment": "calculator",
        "token_ids": list(range(100, 100 + length)),
        "attention_mask": [1] * length,
        "model_generated_mask": model,
        "critical_mask": critical,
        "spans": spans,
        "turns": [Turn(turn_id=0)],
        "tokenizer_fingerprint": "fp-abc",
        "model_id": "toy-student",
        "termination_reason": TerminationReason.FINAL_ANSWER,
    }
    payload.update(overrides)
    return Trajectory(**payload)


# ---------------------------------------------------------------- provenance


def test_only_assistant_spans_are_trainable():
    assert {
        SpanType.ASSISTANT_TEXT,
        SpanType.ASSISTANT_TOOL_CALL,
        SpanType.ASSISTANT_FINAL,
    } == MODEL_GENERATED_SPAN_TYPES
    assert {SpanType.ASSISTANT_TOOL_CALL, SpanType.ASSISTANT_FINAL} == CRITICAL_SPAN_TYPES
    for span_type in (SpanType.SYSTEM, SpanType.USER, SpanType.TOOL_RESULT):
        assert span_type not in MODEL_GENERATED_SPAN_TYPES


def test_masks_are_derived_from_spans_not_trusted():
    traj = _trajectory()
    # tool call 8-11 and final 16-19 are trainable; nothing else is.
    assert traj.model_token_positions() == [8, 9, 10, 11, 16, 17, 18, 19]
    assert traj.critical_token_positions() == [8, 9, 10, 11, 16, 17, 18, 19]
    assert sum(traj.model_generated_mask) == 8
    assert traj.token_counts_by_span_type() == {
        "system": 4,
        "user": 4,
        "assistant_tool_call": 4,
        "tool_result": 4,
        "assistant_final": 4,
    }


def test_tampered_mask_marking_tool_output_trainable_is_rejected():
    """The headline guarantee: a hand-edited file cannot smuggle in tool output."""
    spans = _spans()
    model, critical = build_masks(spans, 20)
    model[13] = True  # a tool_result token
    with pytest.raises(ValidationError, match="model_generated_mask disagrees"):
        _trajectory(spans=spans, model_generated_mask=model, critical_mask=critical)


def test_tampered_critical_mask_is_rejected():
    spans = _spans()
    model, critical = build_masks(spans, 20)
    critical[5] = True  # a user token
    with pytest.raises(ValidationError, match="critical_mask disagrees"):
        _trajectory(spans=spans, model_generated_mask=model, critical_mask=critical)


def test_spans_must_tile_the_sequence():
    gap = [
        (SpanType.SYSTEM, 0, 4),
        (SpanType.ASSISTANT_FINAL, 6, 10),  # gap at 4-5
    ]
    with pytest.raises(ValidationError, match="without gaps or overlaps"):
        _trajectory(spans=_spans(gap))


def test_spans_must_not_overlap():
    overlap = [
        (SpanType.SYSTEM, 0, 6),
        (SpanType.ASSISTANT_FINAL, 4, 10),
    ]
    with pytest.raises(ValidationError, match="without gaps or overlaps"):
        _trajectory(spans=_spans(overlap))


def test_spans_must_cover_every_token():
    short = [(SpanType.SYSTEM, 0, 4), (SpanType.ASSISTANT_FINAL, 4, 8)]
    spans = _spans(short)
    model, critical = build_masks(spans, 8)
    with pytest.raises(ValidationError, match="spans cover 8 tokens"):
        Trajectory(
            trajectory_id="t",
            task_id="k",
            environment="calculator",
            token_ids=list(range(12)),
            attention_mask=[1] * 12,
            model_generated_mask=model + [False] * 4,
            critical_mask=critical + [False] * 4,
            spans=spans,
            turns=[Turn(turn_id=0)],
            tokenizer_fingerprint="fp",
            model_id="m",
            termination_reason=TerminationReason.MAX_TURNS,
        )


def test_mask_length_must_match_token_count():
    spans = _spans()
    model, critical = build_masks(spans, 20)
    with pytest.raises(ValidationError, match="attention_mask has length"):
        _trajectory(
            spans=spans, model_generated_mask=model, critical_mask=critical, attention_mask=[1] * 19
        )


def test_attention_mask_must_be_binary():
    with pytest.raises(ValidationError, match="only 0/1"):
        _trajectory(attention_mask=[2] * 20)


def test_empty_span_is_rejected():
    with pytest.raises(ValidationError, match="empty range"):
        Span(span_type=SpanType.USER, start=3, end=3, turn_id=0)


def test_span_at_and_out_of_range():
    traj = _trajectory()
    assert traj.span_at(9).span_type is SpanType.ASSISTANT_TOOL_CALL
    assert traj.span_at(14).span_type is SpanType.TOOL_RESULT
    with pytest.raises(IndexError):
        traj.span_at(999)


def test_build_masks_rejects_a_span_past_the_end():
    with pytest.raises(TrajectoryError, match="ends at"):
        build_masks(_spans(), 10)


# ------------------------------------------------------- causal alignment


def test_position_zero_can_never_be_a_target():
    """Nothing precedes token 0, so it cannot be supervised."""
    mask = [True] * 6
    assert model_target_positions(mask) == [1, 2, 3, 4, 5]
    with pytest.raises(TrajectoryError, match="position 0"):
        prediction_positions([0, 1])
    with pytest.raises(TrajectoryError, match="position 0"):
        validate_target_positions([0], mask)


def test_prediction_position_is_exactly_target_minus_one():
    assert prediction_positions([1, 5, 19]) == [0, 4, 18]


def test_validate_target_positions_rejects_context_tokens():
    traj = _trajectory()
    for bad in (1, 5, 13):  # system, user, tool_result
        with pytest.raises(TrajectoryError, match="not a model-generated token"):
            validate_target_positions([bad], traj.model_generated_mask)


def test_validate_target_positions_rejects_duplicates_and_disorder():
    traj = _trajectory()
    with pytest.raises(TrajectoryError, match="more than once"):
        validate_target_positions([9, 9], traj.model_generated_mask)
    with pytest.raises(TrajectoryError, match="strictly increasing"):
        validate_target_positions([10, 9], traj.model_generated_mask)
    with pytest.raises(TrajectoryError, match="outside"):
        validate_target_positions([500], traj.model_generated_mask)


def test_critical_target_positions_are_a_subset_of_model_positions():
    traj = _trajectory()
    model = set(model_target_positions(traj.model_generated_mask))
    critical = set(critical_target_positions(traj.model_generated_mask, traj.critical_mask))
    assert critical <= model


def test_positions_by_span_type_counts_correctly():
    traj = _trajectory()
    counts = positions_by_span_type([8, 9, 16, 17, 18], traj.spans)
    assert counts == {"assistant_tool_call": 2, "assistant_final": 3}
    with pytest.raises(TrajectoryError, match="outside the span partition"):
        positions_by_span_type([999], traj.spans)


def test_identity_alignment_shifts_by_exactly_one():
    traj = _trajectory()
    targets = [8, 9, 17]
    alignment = identity_alignment(traj, targets, [1.0, 1.0, 2.0])
    assert alignment.student_prediction_positions == [7, 8, 16]
    assert alignment.teacher_prediction_positions == [7, 8, 16]
    assert alignment.target_token_ids == [traj.token_ids[j] for j in targets]
    assert alignment.is_identity()
    assert alignment.total_weight == pytest.approx(4.0)
    assert alignment.counts_by_span_type() == {"assistant_tool_call": 2, "assistant_final": 1}


def test_alignment_rejects_weight_count_mismatch():
    traj = _trajectory()
    with pytest.raises(AlignmentError, match="weights for"):
        identity_alignment(traj, [8, 9], [1.0])


def test_alignment_rejects_non_model_weight():
    with pytest.raises(ValidationError, match="non-model token was given a non-zero weight"):
        from miniverl.schemas.alignment import AlignmentMap

        AlignmentMap(
            trajectory_id="t",
            student_prediction_positions=[3],
            teacher_prediction_positions=[3],
            target_token_ids=[7],
            model_token_mask=[False],
            token_weights=[1.0],
            span_types=["tool_result"],
        )


def test_alignment_rejects_non_increasing_student_positions():
    from miniverl.schemas.alignment import AlignmentMap

    with pytest.raises(ValidationError, match="strictly increasing"):
        AlignmentMap(
            trajectory_id="t",
            student_prediction_positions=[5, 4],
            teacher_prediction_positions=[5, 4],
            target_token_ids=[1, 2],
            model_token_mask=[True, True],
            token_weights=[1.0, 1.0],
            span_types=["assistant_final", "assistant_final"],
        )


# ------------------------------------------- privileged-context alignment


def _teacher_view(student: Trajectory, prefix_tokens: int) -> Trajectory:
    """Rebuild ``student`` behind an extra system prefix, keeping segment keys."""
    prefix = Span(
        span_type=SpanType.SYSTEM,
        start=0,
        end=prefix_tokens,
        turn_id=0,
        text="privileged",
        metadata={SEGMENT_KEY: "privileged"},
    )
    spans = [prefix]
    cursor = prefix_tokens
    for span in student.spans:
        spans.append(span.model_copy(update={"start": cursor, "end": cursor + span.length}))
        cursor += span.length
    tokens = list(range(900, 900 + prefix_tokens)) + list(student.token_ids)
    model, critical = build_masks(spans, cursor)
    return Trajectory(
        trajectory_id=student.trajectory_id + ":privileged",
        task_id=student.task_id,
        environment=student.environment,
        token_ids=tokens,
        attention_mask=[1] * cursor,
        model_generated_mask=model,
        critical_mask=critical,
        spans=spans,
        turns=list(student.turns),
        tokenizer_fingerprint=student.tokenizer_fingerprint,
        model_id=student.model_id,
        termination_reason=student.termination_reason,
    )


def test_privileged_alignment_applies_the_right_offset():
    student = _trajectory()
    teacher = _teacher_view(student, prefix_tokens=7)
    targets = [8, 9, 17]
    alignment = build_alignment_map(student, targets, [1.0] * 3, teacher=teacher)
    assert alignment.student_prediction_positions == [7, 8, 16]
    assert alignment.teacher_prediction_positions == [14, 15, 23]
    assert not alignment.is_identity()
    # The contract: the *target tokens* are identical on both sides.
    for i, j in enumerate(targets):
        teacher_target = alignment.teacher_prediction_positions[i] + 1
        assert teacher.token_ids[teacher_target] == student.token_ids[j]
        assert alignment.target_token_ids[i] == student.token_ids[j]


def test_privileged_alignment_rejects_a_target_token_mismatch():
    student = _trajectory()
    teacher = _teacher_view(student, prefix_tokens=5)
    corrupted = list(teacher.token_ids)
    corrupted[5 + 9] = 424242
    broken = teacher.model_copy(update={"token_ids": corrupted})
    with pytest.raises(AlignmentError, match="target token mismatch"):
        build_alignment_map(student, [8, 9], [1.0, 1.0], teacher=broken)


def test_privileged_alignment_rejects_a_length_changed_segment():
    student = _trajectory()
    teacher = _teacher_view(student, prefix_tokens=3)
    spans = list(teacher.spans)
    # Shrink the tool-call span by one token and absorb it into the neighbour.
    call_index = next(i for i, s in enumerate(spans) if s.span_type is SpanType.ASSISTANT_TOOL_CALL)
    spans[call_index] = spans[call_index].model_copy(update={"end": spans[call_index].end - 1})
    spans[call_index + 1] = spans[call_index + 1].model_copy(
        update={"start": spans[call_index + 1].start - 1}
    )
    model, critical = build_masks(spans, teacher.length)
    broken = teacher.model_copy(
        update={"spans": spans, "model_generated_mask": model, "critical_mask": critical}
    )
    with pytest.raises(AlignmentError, match="teacher tokens"):
        build_alignment_map(student, [8, 9], [1.0, 1.0], teacher=broken)


def test_privileged_alignment_rejects_a_missing_segment():
    student = _trajectory()
    teacher = _teacher_view(student, prefix_tokens=4)
    spans = [
        s.model_copy(update={"metadata": {SEGMENT_KEY: "renamed"}})
        if s.span_type is SpanType.ASSISTANT_TOOL_CALL
        else s
        for s in teacher.spans
    ]
    broken = teacher.model_copy(update={"spans": spans})
    with pytest.raises(AlignmentError, match="no counterpart in the teacher render"):
        build_alignment_map(student, [8, 9], [1.0, 1.0], teacher=broken)


def test_privileged_alignment_rejects_a_tokenizer_mismatch():
    student = _trajectory()
    teacher = _teacher_view(student, prefix_tokens=4)
    broken = teacher.model_copy(update={"tokenizer_fingerprint": "fp-different"})
    with pytest.raises(TokenizerMismatchError, match="only supports same-tokenizer"):
        build_alignment_map(student, [8], [1.0], teacher=broken)


def test_privileged_alignment_requires_segment_keys():
    student = _trajectory()
    teacher = _teacher_view(student, prefix_tokens=4)
    spans = [s.model_copy(update={"metadata": {}}) for s in teacher.spans]
    broken = teacher.model_copy(update={"spans": spans})
    with pytest.raises(AlignmentError, match="segment_key"):
        build_alignment_map(student, [8], [1.0], teacher=broken)


def test_privileged_alignment_rejects_duplicate_segment_keys():
    student = _trajectory()
    teacher = _teacher_view(student, prefix_tokens=4)
    spans = [s.model_copy(update={"metadata": {SEGMENT_KEY: "same"}}) for s in teacher.spans]
    broken = teacher.model_copy(update={"spans": spans})
    with pytest.raises(AlignmentError, match="duplicate segment key"):
        build_alignment_map(student, [8], [1.0], teacher=broken)


def test_build_alignment_map_without_a_teacher_is_the_identity():
    student = _trajectory()
    a = build_alignment_map(student, [8, 17], [1.0, 1.0])
    b = identity_alignment(student, [8, 17], [1.0, 1.0])
    assert a.model_dump() == b.model_dump()
