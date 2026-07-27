"""Transcript codec: reversibility and token-boundary behaviour.

The codec is what connects text the model produced to token spans the loss
masks.  If it were lossy or if a token could straddle a provenance boundary,
every downstream guarantee would be built on sand.
"""

from __future__ import annotations

import pytest

from miniverl.agent.protocol import (
    FINAL_CLOSE,
    FINAL_OPEN,
    TOOL_CALL_CLOSE,
    TOOL_CALL_OPEN,
    render_final,
    render_tool_call,
    render_tool_result,
)
from miniverl.agent.transcript import (
    ChatFormat,
    Segment,
    TranscriptBuilder,
    token_index_at_char,
)
from miniverl.errors import BackendError, TrajectoryError
from miniverl.models.tokenizers import PROBE_TEXT, ToyTokenizer, tokenizer_fingerprint
from miniverl.schemas.trajectory import SpanType, TerminationReason, Turn
from miniverl.trajectory.alignment import SEGMENT_KEY


@pytest.fixture
def tok() -> ToyTokenizer:
    return ToyTokenizer()


# ------------------------------------------------------------- tokenizer


def test_toy_tokenizer_is_reversible(tok: ToyTokenizer):
    samples = [
        "hello world",
        "<|im_start|>system\nYou are a tool-using assistant.<|im_end|>\n",
        render_tool_call("calculator", {"expression": "2*(3+4)"}),
        render_tool_result(True, result="14"),
        render_tool_result(False, error="no <tool_call> block found."),
        render_final("-3.5"),
        PROBE_TEXT,
        "  \t\n mixed   whitespace \n\n",
        "".join(chr(c) for c in range(32, 127)),
        "0123456789 +-*/()%[]{}\"'\\",
    ]
    for text in samples:
        ids = tok.encode(text)
        assert tok.decode(ids) == text, text[:40]


def test_toy_tokenizer_has_no_cross_boundary_merges(tok: ToyTokenizer):
    """Concatenating segment encodings equals encoding the concatenation."""
    parts = ["<|im_start|>user\n", "Compute 2 + 3.", "<|im_end|>\n", "<final>\n5\n</final>"]
    joined = tok.encode("".join(parts))
    piecewise = [t for part in parts for t in tok.encode(part)]
    assert joined == piecewise


def test_toy_tokenizer_recognises_specials_as_single_tokens(tok: ToyTokenizer):
    for special in (
        "<|im_start|>",
        "<|im_end|>",
        TOOL_CALL_OPEN,
        TOOL_CALL_CLOSE,
        FINAL_OPEN,
        FINAL_CLOSE,
    ):
        assert len(tok.encode(special)) == 1, special


def test_toy_tokenizer_rejects_non_ascii(tok: ToyTokenizer):
    with pytest.raises(BackendError, match="outside the toy tokenizer"):
        tok.encode("café")
    with pytest.raises(BackendError, match="outside the toy tokenizer"):
        tok.encode("汉字")


def test_toy_tokenizer_rejects_out_of_range_ids(tok: ToyTokenizer):
    with pytest.raises(BackendError, match="outside the toy vocabulary"):
        tok.decode([tok.vocab_size])
    with pytest.raises(BackendError, match="outside the toy vocabulary"):
        tok.decode([-1])


def test_toy_tokenizer_fingerprint_is_deterministic():
    assert ToyTokenizer().fingerprint == ToyTokenizer().fingerprint
    assert tokenizer_fingerprint({"a": 1}) != tokenizer_fingerprint({"a": 2})


def test_toy_tokenizer_vocab_and_specials(tok: ToyTokenizer):
    assert 150 < tok.vocab_size < 400
    assert tok.eos_token_id != tok.pad_token_id
    assert tok.token_piece(tok.eos_token_id) == "<|im_end|>"
    assert len(set(tok.vocab)) == tok.vocab_size


# ------------------------------------------------------ boundary splitting


def test_token_index_at_char_finds_clean_boundaries(tok: ToyTokenizer):
    text = "abc" + TOOL_CALL_OPEN + "xyz"
    ids = tok.encode(text)
    index = token_index_at_char(tok, ids, len("abc"))
    assert tok.decode(ids[:index]) == "abc"
    assert tok.decode(ids[index:]).startswith(TOOL_CALL_OPEN)


def test_token_index_at_char_places_a_straddling_token_in_the_typed_span(tok: ToyTokenizer):
    """A token spanning the landmark must go to the later (typed) span."""
    text = "answer" + FINAL_OPEN  # "answer" is one toy token
    ids = tok.encode(text)
    # Split three characters into the middle of the "answer" token.
    index = token_index_at_char(tok, ids, 3)
    assert index == 0
    assert tok.decode(ids[index:]) == text


def test_token_index_at_char_edges(tok: ToyTokenizer):
    ids = tok.encode("hello")
    assert token_index_at_char(tok, ids, 0) == 0
    assert token_index_at_char(tok, ids, 999) == len(ids)


# ----------------------------------------------------------------- builder


def _framed_builder(tok: ToyTokenizer) -> TranscriptBuilder:
    builder = TranscriptBuilder(tok)
    builder.add_context(
        key="sys",
        span_type=SpanType.SYSTEM,
        turn_id=0,
        role="system",
        body="Use tools.",
        open_next_assistant=False,
    )
    builder.add_context(
        key="user",
        span_type=SpanType.USER,
        turn_id=0,
        role="user",
        body="Compute 1 + 1.",
        open_next_assistant=True,
    )
    return builder


def test_context_segments_carry_the_assistant_header(tok: ToyTokenizer):
    """Generation must begin exactly at the first token of a model span."""
    builder = _framed_builder(tok)
    user_span = builder.segments[-1]
    assert user_span.text.endswith("<|im_start|>assistant\n")
    assert user_span.span_type is SpanType.USER  # header is context, never trainable


def test_build_produces_consistent_masks_and_reversible_text(tok: ToyTokenizer):
    builder = _framed_builder(tok)
    builder.add(
        Segment(
            key="gen:0:block",
            span_type=SpanType.ASSISTANT_TOOL_CALL,
            turn_id=0,
            text=render_tool_call("calculator", {"expression": "1 + 1"}),
        )
    )
    builder.add_context(
        key="obs:0",
        span_type=SpanType.TOOL_RESULT,
        turn_id=0,
        role="user",
        body=render_tool_result(True, result="2"),
        close_previous=True,
        open_next_assistant=True,
    )
    builder.add(
        Segment(
            key="gen:1:block",
            span_type=SpanType.ASSISTANT_FINAL,
            turn_id=1,
            text=render_final("2"),
        )
    )
    traj = builder.build(
        trajectory_id="t",
        task_id="k",
        environment="calculator",
        model_id="toy",
        model_revision=None,
        policy_version=3,
        termination_reason=TerminationReason.FINAL_ANSWER,
        turns=[Turn(turn_id=0), Turn(turn_id=1, is_final=True)],
    )
    assert tok.decode(traj.token_ids) == builder.text()
    assert "".join(s.text for s in traj.spans) == builder.text()
    assert traj.policy_version == 3
    assert traj.tokenizer_fingerprint == tok.fingerprint
    trainable = {s.span_type.value for s in traj.spans if s.is_model_generated}
    assert trainable == {"assistant_tool_call", "assistant_final"}
    for span in traj.spans:
        assert span.metadata[SEGMENT_KEY]
    # Provenance: no tool_result token is trainable.
    for span in traj.spans:
        if span.span_type is SpanType.TOOL_RESULT:
            assert not any(traj.model_generated_mask[span.start : span.end])


def test_pretokenized_segments_are_stored_verbatim(tok: ToyTokenizer):
    """Sampled ids must never be re-derived from text."""
    builder = _framed_builder(tok)
    sampled = tok.encode(render_final("7"))
    builder.add(
        Segment(
            key="gen:0:block",
            span_type=SpanType.ASSISTANT_FINAL,
            turn_id=0,
            token_ids=list(sampled),
        )
    )
    traj = builder.build(
        trajectory_id="t",
        task_id="k",
        environment="calculator",
        model_id="toy",
        model_revision=None,
        policy_version=0,
        termination_reason=TerminationReason.FINAL_ANSWER,
        turns=[Turn(turn_id=0)],
    )
    final_span = traj.spans[-1]
    assert traj.token_ids[final_span.start : final_span.end] == sampled


def test_duplicate_segment_keys_are_rejected(tok: ToyTokenizer):
    builder = _framed_builder(tok)
    with pytest.raises(TrajectoryError, match="duplicate transcript segment key"):
        builder.add_context(
            key="sys",
            span_type=SpanType.SYSTEM,
            turn_id=0,
            role="system",
            body="again",
            open_next_assistant=False,
        )


def test_empty_segment_is_rejected(tok: ToyTokenizer):
    builder = TranscriptBuilder(tok)
    with pytest.raises(TrajectoryError, match="is empty"):
        builder.add(Segment(key="x", span_type=SpanType.USER, turn_id=0, text=""))


def test_building_with_no_segments_is_rejected(tok: ToyTokenizer):
    builder = TranscriptBuilder(tok)
    with pytest.raises(TrajectoryError, match="no segments"):
        builder.build(
            trajectory_id="t",
            task_id="k",
            environment="calculator",
            model_id="toy",
            model_revision=None,
            policy_version=0,
            termination_reason=TerminationReason.MAX_TURNS,
        )


def test_chat_format_is_configurable(tok: ToyTokenizer):
    fmt = ChatFormat(turn_start="<|im_start|>", turn_end="<|im_end|>", newline="\n")
    assert fmt.header("system") == "<|im_start|>system\n"
    assert fmt.close() == "<|im_end|>\n"
    builder = TranscriptBuilder(tok, fmt)
    builder.add_context(
        key="s",
        span_type=SpanType.SYSTEM,
        turn_id=0,
        role="system",
        body="x",
        open_next_assistant=False,
    )
    assert builder.text() == "<|im_start|>system\nx<|im_end|>\n"


def test_builder_length_and_token_ids_track_additions(tok: ToyTokenizer):
    builder = _framed_builder(tok)
    before = builder.length
    builder.add(
        Segment(key="g", span_type=SpanType.ASSISTANT_FINAL, turn_id=0, text=render_final("1"))
    )
    assert builder.length > before
    assert len(builder.token_ids) == builder.length
