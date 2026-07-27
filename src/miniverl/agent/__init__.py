"""Agent transcript protocol, tokenized transcript codec and the rollout loop."""

from __future__ import annotations

from miniverl.agent.protocol import (
    FINAL_CLOSE,
    FINAL_OPEN,
    TOOL_CALL_CLOSE,
    TOOL_CALL_OPEN,
    TOOL_RESULT_CLOSE,
    TOOL_RESULT_OPEN,
    ActionKind,
    ParsedAction,
    parse_assistant_text,
    render_final,
    render_tool_call,
    render_tool_result,
    stop_sequences,
)
from miniverl.agent.transcript import ChatFormat, Segment, TranscriptBuilder, token_index_at_char

__all__ = [
    "ActionKind",
    "ParsedAction",
    "parse_assistant_text",
    "render_tool_call",
    "render_tool_result",
    "render_final",
    "stop_sequences",
    "TOOL_CALL_OPEN",
    "TOOL_CALL_CLOSE",
    "TOOL_RESULT_OPEN",
    "TOOL_RESULT_CLOSE",
    "FINAL_OPEN",
    "FINAL_CLOSE",
    "ChatFormat",
    "Segment",
    "TranscriptBuilder",
    "token_index_at_char",
]
