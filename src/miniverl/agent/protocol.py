"""The miniVERL tool-call text protocol.

Deliberately *not* a vendor function-calling format.  Every supported model
emits the same three plain-text blocks, so the same trajectory schema, the same
parser and the same masks work across model families without special-token
surgery or an embedding resize.

::

    <tool_call>
    {"name": "calculator", "arguments": {"expression": "2*(3+4)"}}
    </tool_call>

    <final>
    14
    </final>

and the environment answers with

::

    <tool_result>
    {"ok": true, "result": "14"}
    </tool_result>

The parser is strict: a missing closing tag, non-JSON contents, a missing
``name``, or a non-object ``arguments`` are all parse errors with a specific,
model-readable message rather than a silent fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from miniverl.errors import ToolCallParseError
from miniverl.utils.strict_json import StrictJSONError, strict_json_dumps, strict_json_loads

__all__ = [
    "TOOL_CALL_OPEN",
    "TOOL_CALL_CLOSE",
    "TOOL_RESULT_OPEN",
    "TOOL_RESULT_CLOSE",
    "FINAL_OPEN",
    "FINAL_CLOSE",
    "ActionKind",
    "ParsedAction",
    "parse_assistant_text",
    "render_tool_call",
    "render_tool_result",
    "render_final",
    "stop_sequences",
    "MAX_TOOL_CALL_JSON_CHARS",
]

TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
TOOL_RESULT_OPEN = "<tool_result>"
TOOL_RESULT_CLOSE = "</tool_result>"
FINAL_OPEN = "<final>"
FINAL_CLOSE = "</final>"

#: Hard bound on the JSON payload we will attempt to parse.  Keeps a degenerate
#: rollout from turning into an unbounded ``json.loads`` on megabytes of text.
MAX_TOOL_CALL_JSON_CHARS = 4096


class ActionKind(str, Enum):
    """What the assistant text turned out to be."""

    TOOL_CALL = "tool_call"
    FINAL = "final"
    PARSE_ERROR = "parse_error"


@dataclass(frozen=True)
class ParsedAction:
    """Result of parsing one assistant turn."""

    kind: ActionKind
    #: Text before the recognised block; becomes an ``assistant_text`` span.
    prefix_text: str
    #: Character index in the original text where the recognised block starts.
    block_start: int
    #: Character index just past the recognised block.
    block_end: int
    tool_name: str | None = None
    arguments: dict[str, Any] | None = None
    final_answer: str | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        """``True`` when the text parsed into a tool call or a final answer."""
        return self.kind is not ActionKind.PARSE_ERROR


def stop_sequences() -> list[str]:
    """Strings that end an assistant turn."""
    return [TOOL_CALL_CLOSE, FINAL_CLOSE]


def _parse_error(text: str, message: str) -> ParsedAction:
    return ParsedAction(
        kind=ActionKind.PARSE_ERROR,
        prefix_text=text,
        block_start=len(text),
        block_end=len(text),
        error=message,
    )


def parse_assistant_text(text: str) -> ParsedAction:
    """Parse one assistant turn into a tool call, a final answer, or an error.

    Optional reasoning may precede the action. After the first complete action
    block only whitespace is accepted, matching the one-block prompt contract.
    """
    call_at = text.find(TOOL_CALL_OPEN)
    final_at = text.find(FINAL_OPEN)

    if call_at < 0 and final_at < 0:
        return _parse_error(
            text,
            f"no {TOOL_CALL_OPEN} or {FINAL_OPEN} block found. Emit exactly one of them.",
        )

    use_call = call_at >= 0 and (final_at < 0 or call_at < final_at)

    if use_call:
        body_start = call_at + len(TOOL_CALL_OPEN)
        close_at = text.find(TOOL_CALL_CLOSE, body_start)
        if close_at < 0:
            return _parse_error(text, f"{TOOL_CALL_OPEN} was never closed with {TOOL_CALL_CLOSE}.")
        payload = text[body_start:close_at].strip()
        if len(payload) > MAX_TOOL_CALL_JSON_CHARS:
            return _parse_error(
                text,
                f"tool call JSON is {len(payload)} characters, over the "
                f"{MAX_TOOL_CALL_JSON_CHARS} character limit.",
            )
        try:
            data = strict_json_loads(payload)
        except StrictJSONError as exc:
            return _parse_error(
                text,
                f"tool call payload is not valid JSON under strict rules ({exc}).",
            )
        if not isinstance(data, dict):
            return _parse_error(text, "tool call payload must be a JSON object.")
        name = data.get("name")
        if not isinstance(name, str) or not name:
            return _parse_error(text, "tool call object needs a non-empty string 'name'.")
        arguments = data.get("arguments", {})
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            return _parse_error(text, "tool call 'arguments' must be a JSON object.")
        block_end = close_at + len(TOOL_CALL_CLOSE)
        if text[block_end:].strip():
            return _parse_error(
                text,
                "non-whitespace text appears after the first action block; "
                "emit exactly one block per turn.",
            )
        return ParsedAction(
            kind=ActionKind.TOOL_CALL,
            prefix_text=text[:call_at],
            block_start=call_at,
            block_end=block_end,
            tool_name=name,
            arguments=arguments,
        )

    body_start = final_at + len(FINAL_OPEN)
    close_at = text.find(FINAL_CLOSE, body_start)
    if close_at < 0:
        return _parse_error(text, f"{FINAL_OPEN} was never closed with {FINAL_CLOSE}.")
    block_end = close_at + len(FINAL_CLOSE)
    if text[block_end:].strip():
        return _parse_error(
            text,
            "non-whitespace text appears after the first action block; "
            "emit exactly one block per turn.",
        )
    answer = text[body_start:close_at].strip()
    return ParsedAction(
        kind=ActionKind.FINAL,
        prefix_text=text[:final_at],
        block_start=final_at,
        block_end=block_end,
        final_answer=answer,
    )


def _escape_closing_tags(payload: str) -> str:
    """Make a JSON payload safe to embed between literal closing tags.

    ``"</"`` becomes ``"<\\/"``.  That is valid JSON which decodes to exactly
    the same string, so the round trip is lossless, but the literal
    ``</tool_call>`` / ``</tool_result>`` byte sequence no longer appears and the
    parser cannot be tricked into closing the block early.
    """
    return payload.replace("</", "<\\/")


def render_tool_call(name: str, arguments: dict[str, Any]) -> str:
    """Render a canonical tool-call block (used by oracle traces)."""
    try:
        payload = strict_json_dumps(
            {"name": name, "arguments": arguments},
            ensure_ascii=False,
            sort_keys=True,
        )
    except StrictJSONError as exc:
        raise ToolCallParseError(
            f"tool call cannot be rendered as finite strict JSON: {exc}"
        ) from exc
    return f"{TOOL_CALL_OPEN}\n{_escape_closing_tags(payload)}\n{TOOL_CALL_CLOSE}"


def render_final(answer: str) -> str:
    """Render a canonical final-answer block.

    A final answer is raw text, not JSON, so the protocol has no way to escape a
    literal ``</final>`` inside it.  Rather than emit a block that would parse
    back to a truncated answer, this raises: a silent truncation would corrupt
    an SFT target.
    """
    if FINAL_CLOSE in answer or FINAL_OPEN in answer:
        raise ToolCallParseError(
            f"a final answer cannot contain the literal {FINAL_OPEN!r} or {FINAL_CLOSE!r} marker",
            hint="the plain-text final block has no escape mechanism; return the "
            "answer through a tool result instead",
        )
    return f"{FINAL_OPEN}\n{answer}\n{FINAL_CLOSE}"


def render_tool_result(ok: bool, result: str = "", error: str | None = None) -> str:
    """Render an environment observation block."""
    payload: dict[str, Any] = {"ok": bool(ok)}
    if ok:
        payload["result"] = result
    else:
        payload["error"] = error or "unknown error"
    try:
        body = strict_json_dumps(payload, ensure_ascii=False, sort_keys=True)
    except StrictJSONError as exc:
        raise ToolCallParseError(
            f"tool result cannot be rendered as finite strict JSON: {exc}"
        ) from exc
    return f"{TOOL_RESULT_OPEN}\n{_escape_closing_tags(body)}\n{TOOL_RESULT_CLOSE}"
