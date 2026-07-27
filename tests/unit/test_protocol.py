"""Invariants of the plain-text agent protocol: ``parse_assistant_text`` + renderers.

The whole trajectory pipeline (span provenance, loss masks, oracle traces, the
generation stop condition) is pinned to exactly one surface syntax, so this file
nails that surface down:

* the parser recognises the **first** of the two action markers and reports the
  recognised block as exact character offsets, so ``prefix_text`` /
  ``block_start`` / ``block_end`` can be trusted to slice the raw generation;
* it fails loudly with a specific, model-readable message instead of guessing
  (unclosed tag, non-JSON, non-object payload, missing/blank/non-string
  ``name``, non-object ``arguments``, oversized payload);
* ``ParsedAction.ok`` is false for exactly one kind, ``PARSE_ERROR``;
* the renderers are the inverse of the parser -- whatever ``render_tool_call``
  and ``render_final`` emit parses back to the same call / answer -- and they
  sort JSON keys so an oracle trace is byte-stable regardless of dict insertion
  order;
* ``stop_sequences()`` is the two closing tags, in that order, and a generation
  truncated at either one is already a complete, parseable block.

Pure string invariants: no torch, no I/O, no randomness.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any

import pytest

from miniverl.agent.protocol import (
    FINAL_CLOSE,
    FINAL_OPEN,
    MAX_TOOL_CALL_JSON_CHARS,
    TOOL_CALL_CLOSE,
    TOOL_CALL_OPEN,
    TOOL_RESULT_CLOSE,
    TOOL_RESULT_OPEN,
    ActionKind,
    parse_assistant_text,
    render_final,
    render_tool_call,
    render_tool_result,
    stop_sequences,
)

PROSE = "I should use the calculator here.\n"
CALL_BLOCK = (
    f'{TOOL_CALL_OPEN}\n{{"arguments": {{"expression": "2*(3+4)"}}, '
    f'"name": "calculator"}}\n{TOOL_CALL_CLOSE}'
)
FINAL_BLOCK = f"{FINAL_OPEN}\n14\n{FINAL_CLOSE}"


def _wrap_call(payload: str) -> str:
    """A tool-call block around a raw (possibly malformed) payload."""
    return f"{TOOL_CALL_OPEN}\n{payload}\n{TOOL_CALL_CLOSE}"


def _block_body(text: str, open_tag: str, close_tag: str) -> str:
    """The stripped payload of a single rendered block."""
    assert text.startswith(open_tag), text
    assert text.endswith(close_tag), text
    return text[len(open_tag) : -len(close_tag)].strip()


def _result_body(text: str) -> dict[str, Any]:
    parsed = json.loads(_block_body(text, TOOL_RESULT_OPEN, TOOL_RESULT_CLOSE))
    assert isinstance(parsed, dict)
    return parsed


def _payload_of_length(length: int) -> str:
    """A *valid* tool-call payload whose serialization is exactly ``length`` chars."""
    empty = json.dumps({"arguments": {"x": ""}, "name": "t"}, sort_keys=True)
    pad = length - len(empty)
    if pad < 0:
        raise ValueError(f"cannot build a valid payload shorter than {len(empty)} chars")
    payload = json.dumps({"arguments": {"x": "a" * pad}, "name": "t"}, sort_keys=True)
    assert len(payload) == length
    return payload


# --------------------------------------------------------------------------- #
# constants and stop sequences
# --------------------------------------------------------------------------- #


def test_stop_sequences_are_the_two_closing_tags_in_order() -> None:
    assert stop_sequences() == ["</tool_call>", "</final>"]
    assert stop_sequences() == [TOOL_CALL_CLOSE, FINAL_CLOSE]


def test_stop_sequences_returns_a_fresh_list_each_call() -> None:
    """A caller handing the list to a sampler must not be able to poison it."""
    first = stop_sequences()
    first.append("</oops>")
    first[0] = "mutated"
    assert stop_sequences() == [TOOL_CALL_CLOSE, FINAL_CLOSE]


def test_action_kind_values_are_stable_serializable_strings() -> None:
    assert [kind.value for kind in ActionKind] == ["tool_call", "final", "parse_error"]
    assert ActionKind.TOOL_CALL == "tool_call"
    assert json.dumps({"kind": ActionKind.FINAL}) == '{"kind": "final"}'


def test_no_open_tag_is_hidden_inside_a_closing_tag() -> None:
    """``find(open)`` must never latch onto a closing tag."""
    opens = [TOOL_CALL_OPEN, FINAL_OPEN, TOOL_RESULT_OPEN]
    closes = [TOOL_CALL_CLOSE, FINAL_CLOSE, TOOL_RESULT_CLOSE]
    for open_tag in opens:
        for close_tag in closes:
            assert open_tag not in close_tag, (open_tag, close_tag)


# --------------------------------------------------------------------------- #
# happy paths and exact offsets
# --------------------------------------------------------------------------- #


def test_clean_tool_call() -> None:
    action = parse_assistant_text(CALL_BLOCK)
    assert action.kind is ActionKind.TOOL_CALL
    assert action.ok is True
    assert action.tool_name == "calculator"
    assert action.arguments == {"expression": "2*(3+4)"}
    assert action.final_answer is None
    assert action.error is None
    assert action.prefix_text == ""
    assert action.block_start == 0
    assert action.block_end == len(CALL_BLOCK)


def test_clean_final_answer() -> None:
    action = parse_assistant_text(FINAL_BLOCK)
    assert action.kind is ActionKind.FINAL
    assert action.ok is True
    assert action.final_answer == "14"
    assert action.tool_name is None
    assert action.arguments is None
    assert action.error is None
    assert action.prefix_text == ""
    assert action.block_start == 0
    assert action.block_end == len(FINAL_BLOCK)


def test_prose_before_a_tool_call_is_split_at_exact_offsets() -> None:
    text = f"{PROSE}{CALL_BLOCK}\nand some trailing chatter"
    action = parse_assistant_text(text)
    assert action.kind is ActionKind.TOOL_CALL
    assert action.prefix_text == PROSE
    assert action.block_start == len(PROSE)
    assert action.block_end == len(PROSE) + len(CALL_BLOCK)
    assert text[action.block_start : action.block_end] == CALL_BLOCK
    assert action.tool_name == "calculator"


def test_prose_before_a_final_answer_is_split_at_exact_offsets() -> None:
    prose = "Adding them up now.\n\n"
    text = f"{prose}{FINAL_BLOCK} ignored tail"
    action = parse_assistant_text(text)
    assert action.kind is ActionKind.FINAL
    assert action.prefix_text == prose
    assert action.block_start == len(prose)
    assert action.block_end == len(prose) + len(FINAL_BLOCK)
    assert text[action.block_start : action.block_end] == FINAL_BLOCK
    assert action.final_answer == "14"


def test_prefix_and_block_partition_the_text_without_overlap() -> None:
    text = f"{PROSE}{CALL_BLOCK}tail"
    action = parse_assistant_text(text)
    assert text[: action.block_start] == action.prefix_text
    assert text == action.prefix_text + text[action.block_start : action.block_end] + "tail"


@pytest.mark.parametrize(
    ("text", "close_tag"),
    [
        (CALL_BLOCK, TOOL_CALL_CLOSE),
        (PROSE + CALL_BLOCK, TOOL_CALL_CLOSE),
        (PROSE + CALL_BLOCK + "\nchatter", TOOL_CALL_CLOSE),
        (FINAL_BLOCK, FINAL_CLOSE),
        ("prose\n" + FINAL_BLOCK + "\nchatter", FINAL_CLOSE),
        (f"{FINAL_OPEN}\n{CALL_BLOCK}\n{FINAL_CLOSE}", FINAL_CLOSE),
    ],
)
def test_block_end_lands_just_past_the_first_closing_tag(text: str, close_tag: str) -> None:
    action = parse_assistant_text(text)
    assert action.ok, action.error
    open_tag = TOOL_CALL_OPEN if close_tag == TOOL_CALL_CLOSE else FINAL_OPEN
    block = text[action.block_start : action.block_end]
    assert block.startswith(open_tag)
    assert block.endswith(close_tag)
    # "just past" means the closing tag occurs exactly once, at the very end.
    assert close_tag not in block[: -len(close_tag)]


# --------------------------------------------------------------------------- #
# first marker wins
# --------------------------------------------------------------------------- #


def test_tool_call_wins_when_it_appears_first() -> None:
    text = f"{CALL_BLOCK}\n{FINAL_BLOCK}"
    action = parse_assistant_text(text)
    assert action.kind is ActionKind.TOOL_CALL
    assert action.tool_name == "calculator"
    assert action.final_answer is None
    assert action.block_start == 0
    assert action.block_end == len(CALL_BLOCK)


def test_final_wins_when_it_appears_first() -> None:
    text = f"{FINAL_BLOCK}\n{CALL_BLOCK}"
    action = parse_assistant_text(text)
    assert action.kind is ActionKind.FINAL
    assert action.final_answer == "14"
    assert action.tool_name is None
    assert action.block_start == 0
    assert action.block_end == len(FINAL_BLOCK)


def test_a_tool_call_nested_inside_a_final_block_stays_part_of_the_answer() -> None:
    text = f"{FINAL_OPEN}\n{CALL_BLOCK}\n{FINAL_CLOSE}"
    action = parse_assistant_text(text)
    assert action.kind is ActionKind.FINAL
    assert action.final_answer == CALL_BLOCK
    assert action.block_end == len(text)


def test_a_final_tag_inside_the_tool_call_json_does_not_steal_the_turn() -> None:
    arguments = {"query": f"what does {FINAL_OPEN} mean?"}
    text = render_tool_call("search", arguments)
    assert FINAL_OPEN in text
    action = parse_assistant_text(text)
    assert action.kind is ActionKind.TOOL_CALL
    assert action.arguments == arguments


def test_a_malformed_first_marker_is_not_rescued_by_a_valid_later_block() -> None:
    """Strict parser: first marker wins even when the second block is perfect."""
    text = f"{FINAL_OPEN} oops I forgot to close this\n{CALL_BLOCK}"
    action = parse_assistant_text(text)
    assert action.kind is ActionKind.PARSE_ERROR
    assert action.error == f"{FINAL_OPEN} was never closed with {FINAL_CLOSE}."


# --------------------------------------------------------------------------- #
# missing / unclosed markers
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text",
    [
        "",
        "just some prose",
        "</tool_call> stray close tag",
        "</final>",
        '<tool_result>\n{"ok": true, "result": "14"}\n</tool_result>',
        "<TOOL_CALL>{}</TOOL_CALL>",
    ],
)
def test_text_without_either_marker_is_a_parse_error(text: str) -> None:
    action = parse_assistant_text(text)
    assert action.kind is ActionKind.PARSE_ERROR
    assert action.error == (
        f"no {TOOL_CALL_OPEN} or {FINAL_OPEN} block found. Emit exactly one of them."
    )


@pytest.mark.parametrize(
    "text",
    [
        TOOL_CALL_OPEN,
        f'{TOOL_CALL_OPEN}\n{{"name": "calculator", "arguments": {{}}}}',
        f'{TOOL_CALL_OPEN}\n{{"name": "calculator"}}\n<tool_call>',
        f"prose\n{TOOL_CALL_OPEN}{{}}",
    ],
)
def test_unclosed_tool_call_is_a_parse_error(text: str) -> None:
    action = parse_assistant_text(text)
    assert action.kind is ActionKind.PARSE_ERROR
    assert action.error == f"{TOOL_CALL_OPEN} was never closed with {TOOL_CALL_CLOSE}."


@pytest.mark.parametrize(
    "text",
    [
        FINAL_OPEN,
        f"{FINAL_OPEN}\n14",
        f"{FINAL_OPEN}\n14\n<final>",
        f"prose\n{FINAL_OPEN}14",
    ],
)
def test_unclosed_final_is_a_parse_error(text: str) -> None:
    action = parse_assistant_text(text)
    assert action.kind is ActionKind.PARSE_ERROR
    assert action.error == f"{FINAL_OPEN} was never closed with {FINAL_CLOSE}."


def test_a_closing_tag_before_its_opening_tag_does_not_count() -> None:
    text = f'{TOOL_CALL_CLOSE}\n{TOOL_CALL_OPEN}\n{{"name": "t"}}'
    action = parse_assistant_text(text)
    assert action.kind is ActionKind.PARSE_ERROR
    assert action.error == f"{TOOL_CALL_OPEN} was never closed with {TOOL_CALL_CLOSE}."


# --------------------------------------------------------------------------- #
# payload validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "payload",
    [
        "",
        "not json at all",
        "{name: 'calculator'}",
        '{"name": "calculator",}',
        '{"name": "calculator"',
        "{'name': 'calculator'}",
        '{"name": "calculator"} trailing',
    ],
)
def test_non_json_payload_is_a_parse_error(payload: str) -> None:
    action = parse_assistant_text(_wrap_call(payload))
    assert action.kind is ActionKind.PARSE_ERROR
    assert action.error is not None
    assert action.error.startswith("tool call payload is not valid JSON (")
    assert action.error.endswith(").")
    assert "at column " in action.error


@pytest.mark.parametrize(
    "payload",
    ["[]", '[{"name": "calculator"}]', '"calculator"', "42", "-3.5", "true", "false", "null"],
)
def test_non_object_json_payload_is_a_parse_error(payload: str) -> None:
    action = parse_assistant_text(_wrap_call(payload))
    assert action.kind is ActionKind.PARSE_ERROR
    assert action.error == "tool call payload must be a JSON object."


@pytest.mark.parametrize(
    "payload",
    [
        "{}",
        '{"arguments": {"expression": "1+1"}}',
        '{"tool_name": "calculator"}',
        '{"name": ""}',
        '{"name": 7}',
        '{"name": 1.5}',
        '{"name": null}',
        '{"name": true}',
        '{"name": ["calculator"]}',
        '{"name": {"tool": "calculator"}}',
    ],
)
def test_missing_blank_or_non_string_name_is_a_parse_error(payload: str) -> None:
    action = parse_assistant_text(_wrap_call(payload))
    assert action.kind is ActionKind.PARSE_ERROR
    assert action.error == "tool call object needs a non-empty string 'name'."


@pytest.mark.parametrize(
    "payload",
    [
        '{"name": "calculator", "arguments": []}',
        '{"name": "calculator", "arguments": ["1+1"]}',
        '{"name": "calculator", "arguments": "expression=1+1"}',
        '{"name": "calculator", "arguments": ""}',
        '{"name": "calculator", "arguments": 5}',
        '{"name": "calculator", "arguments": true}',
    ],
)
def test_non_object_arguments_is_a_parse_error(payload: str) -> None:
    action = parse_assistant_text(_wrap_call(payload))
    assert action.kind is ActionKind.PARSE_ERROR
    assert action.error == "tool call 'arguments' must be a JSON object."


@pytest.mark.parametrize(
    "payload",
    [
        '{"name": "clock"}',
        '{"name": "clock", "arguments": null}',
        '{"name": "clock", "arguments": {}}',
    ],
)
def test_absent_null_or_empty_arguments_default_to_an_empty_dict(payload: str) -> None:
    action = parse_assistant_text(_wrap_call(payload))
    assert action.kind is ActionKind.TOOL_CALL, action.error
    assert action.tool_name == "clock"
    assert action.arguments == {}


def test_defaulted_arguments_are_a_fresh_dict_per_call() -> None:
    """A shared default would let one rollout mutate another's arguments."""
    first = parse_assistant_text(_wrap_call('{"name": "clock"}'))
    second = parse_assistant_text(_wrap_call('{"name": "clock"}'))
    assert first.arguments is not None
    assert first.arguments is not second.arguments
    first.arguments["injected"] = True
    assert second.arguments == {}


def test_payload_exactly_at_the_char_limit_is_accepted() -> None:
    payload = _payload_of_length(MAX_TOOL_CALL_JSON_CHARS)
    action = parse_assistant_text(_wrap_call(payload))
    assert action.kind is ActionKind.TOOL_CALL, action.error
    assert action.arguments is not None
    assert len(action.arguments["x"]) == MAX_TOOL_CALL_JSON_CHARS - 37


def test_payload_one_char_over_the_limit_is_rejected() -> None:
    over = MAX_TOOL_CALL_JSON_CHARS + 1
    action = parse_assistant_text(_wrap_call(_payload_of_length(over)))
    assert action.kind is ActionKind.PARSE_ERROR
    assert action.error == (
        f"tool call JSON is {over} characters, over the {MAX_TOOL_CALL_JSON_CHARS} character limit."
    )


def test_the_size_guard_runs_before_json_parsing() -> None:
    """The point of the bound is to *avoid* json.loads on a degenerate rollout."""
    payload = "x" * (MAX_TOOL_CALL_JSON_CHARS + 100)
    action = parse_assistant_text(_wrap_call(payload))
    assert action.kind is ActionKind.PARSE_ERROR
    assert "over the" in (action.error or "")
    assert "not valid JSON" not in (action.error or "")


def test_the_size_limit_is_measured_after_stripping_whitespace() -> None:
    payload = _payload_of_length(MAX_TOOL_CALL_JSON_CHARS)
    padding = "\n" * 64
    action = parse_assistant_text(f"{TOOL_CALL_OPEN}{padding}{payload}{padding}{TOOL_CALL_CLOSE}")
    assert action.kind is ActionKind.TOOL_CALL, action.error


def test_the_final_block_has_no_size_limit() -> None:
    answer = "9" * (MAX_TOOL_CALL_JSON_CHARS * 3)
    action = parse_assistant_text(render_final(answer))
    assert action.kind is ActionKind.FINAL
    assert action.final_answer == answer


# --------------------------------------------------------------------------- #
# whitespace and unicode
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("pad", ["", " ", "\n", "\n\n", "  \t ", "\r\n", "\n   \t\r\n"])
def test_whitespace_around_the_tool_call_payload_is_ignored(pad: str) -> None:
    payload = '{"name": "calculator", "arguments": {"expression": "1+1"}}'
    text = f"{TOOL_CALL_OPEN}{pad}{payload}{pad}{TOOL_CALL_CLOSE}"
    action = parse_assistant_text(text)
    assert action.kind is ActionKind.TOOL_CALL, action.error
    assert action.tool_name == "calculator"
    assert action.arguments == {"expression": "1+1"}
    assert action.block_end == len(text)


def test_a_pretty_printed_multiline_payload_parses() -> None:
    payload = json.dumps({"name": "calculator", "arguments": {"expression": "1+1"}}, indent=2)
    assert "\n" in payload
    action = parse_assistant_text(_wrap_call(payload))
    assert action.kind is ActionKind.TOOL_CALL, action.error
    assert action.arguments == {"expression": "1+1"}


@pytest.mark.parametrize("pad", ["", " ", "\n", "\n\n\t", "\r\n"])
def test_whitespace_around_the_final_answer_is_stripped(pad: str) -> None:
    text = f"{FINAL_OPEN}{pad}14{pad}{FINAL_CLOSE}"
    action = parse_assistant_text(text)
    assert action.kind is ActionKind.FINAL
    assert action.final_answer == "14"
    assert action.block_end == len(text)


def test_interior_newlines_of_a_final_answer_survive() -> None:
    answer = "line one\nline two\n\nline four"
    action = parse_assistant_text(f"{FINAL_OPEN}\n{answer}\n{FINAL_CLOSE}")
    assert action.final_answer == answer


def test_an_empty_final_block_yields_an_empty_answer() -> None:
    action = parse_assistant_text(f"{FINAL_OPEN}\n \n{FINAL_CLOSE}")
    assert action.kind is ActionKind.FINAL
    assert action.final_answer == ""
    assert action.ok is True


def test_unicode_inside_the_tool_call_json_survives_the_round_trip() -> None:
    arguments = {"expression": "π ≈ 3.14", "note": "日本語のテキスト", "mood": "🙂"}
    action = parse_assistant_text(render_tool_call("計算器", arguments))
    assert action.kind is ActionKind.TOOL_CALL, action.error
    assert action.tool_name == "計算器"
    assert action.arguments == arguments


def test_unicode_escapes_in_the_payload_decode_to_the_same_text() -> None:
    payload = json.dumps({"name": "t", "arguments": {"q": "π 🙂"}}, ensure_ascii=True)
    assert "\\u" in payload
    action = parse_assistant_text(_wrap_call(payload))
    assert action.kind is ActionKind.TOOL_CALL, action.error
    assert action.arguments == {"q": "π 🙂"}


def test_unicode_final_answer_survives_the_round_trip() -> None:
    answer = "答案是 14 — π ≈ 3.14 🙂"
    action = parse_assistant_text(render_final(answer))
    assert action.final_answer == answer


def test_offsets_are_character_indices_not_byte_offsets() -> None:
    prose = "🙂🙂 thinking\n"
    text = prose + FINAL_BLOCK
    action = parse_assistant_text(text)
    assert action.block_start == len(prose)
    assert len(prose.encode()) != len(prose)
    assert text[action.block_start : action.block_end] == FINAL_BLOCK


# --------------------------------------------------------------------------- #
# ParsedAction contract
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("text", "kind", "ok"),
    [
        (CALL_BLOCK, ActionKind.TOOL_CALL, True),
        (PROSE + CALL_BLOCK, ActionKind.TOOL_CALL, True),
        (FINAL_BLOCK, ActionKind.FINAL, True),
        (f"{FINAL_OPEN}\n\n{FINAL_CLOSE}", ActionKind.FINAL, True),
        ("no markers here", ActionKind.PARSE_ERROR, False),
        (TOOL_CALL_OPEN + "{}", ActionKind.PARSE_ERROR, False),
        (_wrap_call("nope"), ActionKind.PARSE_ERROR, False),
        (_wrap_call("[]"), ActionKind.PARSE_ERROR, False),
        (_wrap_call('{"name": ""}'), ActionKind.PARSE_ERROR, False),
        (_wrap_call('{"name": "t", "arguments": 1}'), ActionKind.PARSE_ERROR, False),
        (FINAL_OPEN + "unclosed", ActionKind.PARSE_ERROR, False),
    ],
)
def test_ok_is_false_exactly_for_parse_error(text: str, kind: ActionKind, ok: bool) -> None:
    action = parse_assistant_text(text)
    assert action.kind is kind
    assert action.ok is ok
    assert (action.error is None) is ok


@pytest.mark.parametrize(
    "text",
    [
        "",
        "no markers here",
        TOOL_CALL_OPEN + "{}",
        _wrap_call("nope"),
        _wrap_call('{"name": 3}'),
        f"{PROSE}{TOOL_CALL_OPEN} unclosed",
    ],
)
def test_parse_errors_hand_back_the_whole_text_as_prefix(text: str) -> None:
    action = parse_assistant_text(text)
    assert action.kind is ActionKind.PARSE_ERROR
    assert action.prefix_text == text
    assert action.block_start == len(text)
    assert action.block_end == len(text)
    assert action.tool_name is None
    assert action.arguments is None
    assert action.final_answer is None
    assert action.error


def test_parsed_action_is_frozen() -> None:
    action = parse_assistant_text(CALL_BLOCK)
    with pytest.raises(dataclasses.FrozenInstanceError):
        action.tool_name = "other"  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# renderers
# --------------------------------------------------------------------------- #

ROUND_TRIP_CALLS: list[tuple[str, dict[str, Any]]] = [
    ("calculator", {"expression": "2*(3+4)"}),
    ("search", {}),
    ("noop", {"flag": False, "nothing": None}),
    ("nested", {"outer": {"inner": [1, 2, {"deep": "value"}]}, "count": 3}),
    ("numbers", {"x": 0.125, "y": -7, "big": 10**12}),
    ("unicode", {"q": "π ≈ 3.14 — 日本語 🙂"}),
    ("blank_value", {"q": ""}),
    ("mentions_tags", {"q": f"is {FINAL_OPEN} or {TOOL_RESULT_OPEN} allowed?"}),
    ("newlines", {"code": "def f():\n    return 1\n"}),
    ("quotes", {"q": 'he said "hi" \\ then left'}),
]


@pytest.mark.parametrize(
    ("name", "arguments"), ROUND_TRIP_CALLS, ids=[c[0] for c in ROUND_TRIP_CALLS]
)
def test_render_tool_call_round_trips(name: str, arguments: dict[str, Any]) -> None:
    text = render_tool_call(name, arguments)
    action = parse_assistant_text(text)
    assert action.kind is ActionKind.TOOL_CALL, action.error
    assert action.tool_name == name
    assert action.arguments == arguments
    assert action.prefix_text == ""
    assert action.block_start == 0
    assert action.block_end == len(text)


def test_render_tool_call_round_trips_after_prose() -> None:
    text = PROSE + render_tool_call("calculator", {"expression": "1+1"})
    action = parse_assistant_text(text)
    assert action.kind is ActionKind.TOOL_CALL, action.error
    assert action.prefix_text == PROSE
    assert action.block_start == len(PROSE)
    assert action.block_end == len(text)


def test_render_tool_call_layout_is_tag_newline_payload_newline_tag() -> None:
    text = render_tool_call("calculator", {"expression": "1+1"})
    assert text.startswith(TOOL_CALL_OPEN + "\n")
    assert text.endswith("\n" + TOOL_CALL_CLOSE)
    lines = text.split("\n")
    assert len(lines) == 3
    assert json.loads(lines[1]) == {"arguments": {"expression": "1+1"}, "name": "calculator"}


def test_render_tool_call_sorts_keys_at_every_depth() -> None:
    text = render_tool_call("zeta", {"b": 1, "a": {"n": 2, "m": 3}})
    payload = _block_body(text, TOOL_CALL_OPEN, TOOL_CALL_CLOSE)
    assert payload == '{"arguments": {"a": {"m": 3, "n": 2}, "b": 1}, "name": "zeta"}'


def test_render_tool_call_is_insertion_order_independent() -> None:
    """Byte-stable oracle traces: two equal dicts must render identically."""
    first = render_tool_call("t", {"x": 1, "y": {"a": 1, "b": 2}})
    second = render_tool_call("t", {"y": {"b": 2, "a": 1}, "x": 1})
    assert first == second


def test_render_tool_call_keeps_unicode_literal() -> None:
    text = render_tool_call("t", {"q": "日本語 π"})
    assert "日本語 π" in text
    assert "\\u" not in text


@pytest.mark.parametrize(
    "answer",
    [
        "14",
        "",
        "-0.5",
        "the answer is 14",
        "line one\nline two",
        "π ≈ 3.14 🙂",
        "  ",
        "</tool_call>",
    ],
)
def test_render_final_round_trips(answer: str) -> None:
    text = render_final(answer)
    action = parse_assistant_text(text)
    assert action.kind is ActionKind.FINAL
    assert action.final_answer == answer.strip()
    assert action.prefix_text == ""
    assert action.block_start == 0
    assert action.block_end == len(text)


def test_render_final_layout_is_tag_newline_answer_newline_tag() -> None:
    text = render_final("14")
    assert text == f"{FINAL_OPEN}\n14\n{FINAL_CLOSE}"


def test_render_final_round_trips_an_answer_mentioning_the_tool_call_tag() -> None:
    answer = f"you should emit {TOOL_CALL_OPEN} to call a tool"
    action = parse_assistant_text(render_final(answer))
    assert action.kind is ActionKind.FINAL
    assert action.final_answer == answer


def test_render_final_only_strips_surrounding_whitespace() -> None:
    action = parse_assistant_text(render_final("   14 \t\n"))
    assert action.final_answer == "14"


def test_render_final_round_trips_an_answer_containing_the_closing_tag() -> None:
    """The plain-text final block has no escape, so rendering must refuse.

    Emitting a block that parses back to a *truncated* answer would silently
    corrupt an SFT target, so render_final raises instead.
    """
    from miniverl.errors import ToolCallParseError

    answer = f"the {FINAL_CLOSE} tag ends a turn"
    with pytest.raises(ToolCallParseError, match=r"no escape mechanism|cannot contain"):
        render_final(answer)
    return
    action = parse_assistant_text(render_final(answer))
    assert action.kind is ActionKind.FINAL
    assert action.final_answer == answer


def test_render_tool_call_round_trips_arguments_containing_the_closing_tag() -> None:
    arguments = {"snippet": f"emit {TOOL_CALL_CLOSE} to stop"}
    action = parse_assistant_text(render_tool_call("echo", arguments))
    assert action.kind is ActionKind.TOOL_CALL, action.error
    assert action.arguments == arguments


# --------------------------------------------------------------------------- #
# render_tool_result
# --------------------------------------------------------------------------- #


def test_render_tool_result_success_is_ok_plus_result() -> None:
    text = render_tool_result(True, result="14")
    assert text.startswith(TOOL_RESULT_OPEN + "\n")
    assert text.endswith("\n" + TOOL_RESULT_CLOSE)
    body = _block_body(text, TOOL_RESULT_OPEN, TOOL_RESULT_CLOSE)
    assert json.loads(body) == {"ok": True, "result": "14"}
    assert body == '{"ok": true, "result": "14"}'


def test_render_tool_result_failure_is_ok_plus_error() -> None:
    body = _block_body(render_tool_result(False, error="boom"), TOOL_RESULT_OPEN, TOOL_RESULT_CLOSE)
    assert json.loads(body) == {"ok": False, "error": "boom"}
    assert body == '{"error": "boom", "ok": false}'


def test_render_tool_result_never_mixes_result_and_error() -> None:
    good = _result_body(render_tool_result(True, result="14", error="ignored"))
    assert set(good) == {"ok", "result"}
    bad = _result_body(render_tool_result(False, result="ignored", error="boom"))
    assert set(bad) == {"ok", "error"}


def test_render_tool_result_defaults_result_to_empty_string() -> None:
    assert _result_body(render_tool_result(True)) == {"ok": True, "result": ""}


@pytest.mark.parametrize("error", [None, ""])
def test_render_tool_result_substitutes_a_default_error_message(error: str | None) -> None:
    assert _result_body(render_tool_result(False, error=error)) == {
        "ok": False,
        "error": "unknown error",
    }


@pytest.mark.parametrize(("ok", "expected"), [(True, True), (False, False), (1, True), (0, False)])
def test_render_tool_result_coerces_ok_to_a_json_bool(ok: object, expected: bool) -> None:
    body = _block_body(
        render_tool_result(bool(ok), result="14", error="boom"),
        TOOL_RESULT_OPEN,
        TOOL_RESULT_CLOSE,
    )
    payload = json.loads(body)
    assert payload["ok"] is expected
    assert '"ok": true' in body or '"ok": false' in body


def test_render_tool_result_keeps_unicode_literal() -> None:
    body = _block_body(
        render_tool_result(True, result="π ≈ 3.14"), TOOL_RESULT_OPEN, TOOL_RESULT_CLOSE
    )
    assert "π ≈ 3.14" in body
    assert "\\u" not in body
    assert json.loads(body)["result"] == "π ≈ 3.14"


@pytest.mark.parametrize(
    "text",
    [
        render_tool_result(True, result="14"),
        render_tool_result(False, error="boom"),
    ],
)
def test_a_tool_result_block_is_never_mistaken_for_an_assistant_action(text: str) -> None:
    action = parse_assistant_text(text)
    assert action.kind is ActionKind.PARSE_ERROR
    assert action.error is not None
    assert TOOL_CALL_OPEN in action.error
    assert FINAL_OPEN in action.error


# --------------------------------------------------------------------------- #
# stop sequences terminate a complete block
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("text", "stop"),
    [
        (render_tool_call("calculator", {"expression": "1+1"}), TOOL_CALL_CLOSE),
        (render_final("14"), FINAL_CLOSE),
        (PROSE + render_tool_call("clock", {}), TOOL_CALL_CLOSE),
    ],
)
def test_a_generation_truncated_at_a_stop_sequence_is_already_complete(
    text: str, stop: str
) -> None:
    assert stop in stop_sequences()
    assert text.endswith(stop)
    action = parse_assistant_text(text)
    assert action.ok, action.error
    assert action.block_end == len(text)
