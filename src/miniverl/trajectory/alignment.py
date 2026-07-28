"""Build and verify student/teacher alignment maps.

Two teacher context modes are supported and they are aligned very differently.

``standard``
    The teacher sees the byte-identical transcript.  Positions coincide, so the
    alignment is the identity and only needs a sanity check.

``privileged_context``
    The teacher additionally sees an oracle block the student never saw, so the
    teacher sequence is *longer* and every shared position is shifted.  The
    shift is not assumed to be constant: each span carries a stable
    ``segment_key`` in its metadata, spans are matched by that key, and the
    offset is computed per span.

In both modes the alignment is only accepted if the **target token ids are
identical** on both sides.  That is what makes the same-tokenizer contract
enforceable rather than aspirational.
"""

from __future__ import annotations

from collections.abc import Sequence

from miniverl.errors import AlignmentError, TokenizerMismatchError
from miniverl.schemas.alignment import AlignmentMap
from miniverl.schemas.trajectory import Span, Trajectory
from miniverl.trajectory.masks import validate_target_positions

__all__ = ["identity_alignment", "build_alignment_map", "SEGMENT_KEY"]

#: Metadata key holding a span's stable identity across student/teacher renders.
SEGMENT_KEY = "segment_key"


def _segment_key(span: Span) -> str:
    key = span.metadata.get(SEGMENT_KEY)
    if not isinstance(key, str):
        raise AlignmentError(
            f"span {span.span_type.value} at [{span.start},{span.end}) has no "
            f"'{SEGMENT_KEY}' metadata; privileged-context alignment needs stable "
            "segment keys produced by the transcript builder",
            hint="rebuild the trajectory with miniverl.agent.transcript.TranscriptBuilder",
        )
    return key


def identity_alignment(
    trajectory: Trajectory,
    target_positions: Sequence[int],
    weights: Sequence[float],
) -> AlignmentMap:
    """Alignment for a teacher that sees exactly the student's context."""
    validate_target_positions(target_positions, trajectory.model_generated_mask)
    if len(weights) != len(target_positions):
        raise AlignmentError(
            f"got {len(weights)} weights for {len(target_positions)} target positions"
        )
    predictions = [j - 1 for j in target_positions]
    return AlignmentMap(
        trajectory_id=trajectory.trajectory_id,
        student_prediction_positions=predictions,
        teacher_prediction_positions=list(predictions),
        target_token_ids=[trajectory.token_ids[j] for j in target_positions],
        model_token_mask=[True] * len(target_positions),
        token_weights=[float(w) for w in weights],
        span_types=[trajectory.span_at(j).span_type.value for j in target_positions],
    )


def build_alignment_map(
    student: Trajectory,
    target_positions: Sequence[int],
    weights: Sequence[float],
    teacher: Trajectory | None = None,
) -> AlignmentMap:
    """Align selected student targets onto teacher prediction positions.

    Parameters
    ----------
    student:
        The rollout the policy actually produced.
    target_positions:
        Selected *target* positions (not prediction positions) in student space.
    weights:
        Per-position loss weights, same length as ``target_positions``.
    teacher:
        The teacher-side render.  ``None`` (or an identical token sequence)
        means ``standard`` mode and yields the identity alignment.
    """
    if teacher is None:
        return identity_alignment(student, target_positions, weights)

    if teacher.tokenizer_fingerprint != student.tokenizer_fingerprint:
        raise TokenizerMismatchError(
            "student and teacher tokenizers differ "
            f"({student.tokenizer_fingerprint[:12]}... vs "
            f"{teacher.tokenizer_fingerprint[:12]}...); miniVERL only supports "
            "same-tokenizer distillation",
            hint="pick a teacher from the same model family, or wait for "
            "cross-tokenizer support (roadmap item, see docs/limitations.md)",
        )

    validate_target_positions(target_positions, student.model_generated_mask)
    if len(weights) != len(target_positions):
        raise AlignmentError(
            f"got {len(weights)} weights for {len(target_positions)} target positions"
        )

    teacher_spans: dict[str, Span] = {}
    for span in teacher.spans:
        key = _segment_key(span)
        if key in teacher_spans:
            raise AlignmentError(
                f"teacher render contains duplicate segment key {key!r}; keys must be unique"
            )
        teacher_spans[key] = span

    student_predictions: list[int] = []
    teacher_predictions: list[int] = []
    target_ids: list[int] = []
    span_types: list[str] = []

    for j in target_positions:
        span = student.span_at(j)
        key = _segment_key(span)
        tspan = teacher_spans.get(key)
        if tspan is None:
            raise AlignmentError(
                f"student span {key!r} has no counterpart in the teacher render; "
                "the privileged-context builder must preserve every student segment"
            )
        if tspan.length != span.length:
            raise AlignmentError(
                f"segment {key!r} has {span.length} student tokens but "
                f"{tspan.length} teacher tokens; the shared content must tokenize "
                "identically on both sides"
            )
        offset = j - span.start
        teacher_j = tspan.start + offset
        student_token = student.token_ids[j]
        teacher_token = teacher.token_ids[teacher_j]
        if student_token != teacher_token:
            raise AlignmentError(
                f"target token mismatch at student position {j} / teacher position "
                f"{teacher_j}: {student_token} != {teacher_token}"
            )
        if teacher_j == 0:
            raise AlignmentError(
                "a target token landed at teacher position 0, which has no "
                "preceding prediction position"
            )
        student_predictions.append(j - 1)
        teacher_predictions.append(teacher_j - 1)
        target_ids.append(student_token)
        span_types.append(span.span_type.value)

    return AlignmentMap(
        trajectory_id=student.trajectory_id,
        student_prediction_positions=student_predictions,
        teacher_prediction_positions=teacher_predictions,
        target_token_ids=target_ids,
        model_token_mask=[True] * len(target_ids),
        token_weights=[float(w) for w in weights],
        span_types=span_types,
    )
