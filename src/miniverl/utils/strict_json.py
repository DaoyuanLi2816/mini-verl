"""Bounded, standards-compliant JSON for untrusted model protocol data."""

from __future__ import annotations

import json
import math
from typing import Any

__all__ = [
    "MAX_JSON_DEPTH",
    "MAX_JSON_INTEGER_DIGITS",
    "MAX_JSON_MEMBERS",
    "StrictJSONError",
    "strict_json_dumps",
    "strict_json_loads",
]

MAX_JSON_DEPTH = 32
MAX_JSON_MEMBERS = 256
MAX_JSON_INTEGER_DIGITS = 128
MAX_JSON_NUMBER_CHARS = 128


class StrictJSONError(ValueError):
    """JSON is non-standard, non-finite, or exceeds a protocol safety bound."""


def _reject_constant(literal: str) -> Any:
    raise StrictJSONError(f"JSON number {literal!r} is not finite")


def _parse_float(literal: str) -> float:
    if len(literal) > MAX_JSON_NUMBER_CHARS:
        raise StrictJSONError(
            f"JSON float literal exceeds the {MAX_JSON_NUMBER_CHARS}-character limit"
        )
    try:
        value = float(literal)
    except (OverflowError, ValueError) as exc:
        raise StrictJSONError("JSON float literal is invalid") from exc
    if not math.isfinite(value):
        raise StrictJSONError("JSON float literal does not produce a finite value")
    return value


def _parse_int(literal: str) -> int:
    digits = literal.lstrip("+-")
    if len(digits) > MAX_JSON_INTEGER_DIGITS:
        raise StrictJSONError(
            f"JSON integer literal exceeds the {MAX_JSON_INTEGER_DIGITS}-digit limit"
        )
    try:
        return int(literal)
    except (OverflowError, ValueError) as exc:
        raise StrictJSONError("JSON integer literal is invalid") from exc


def _object_from_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StrictJSONError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _validate_string(value: str) -> None:
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise StrictJSONError("JSON text contains an invalid Unicode surrogate")


def _validate(value: Any, *, depth: int, members: list[int]) -> None:
    if depth > MAX_JSON_DEPTH:
        raise StrictJSONError(f"JSON container depth exceeds the {MAX_JSON_DEPTH}-level limit")
    if isinstance(value, str):
        _validate_string(value)
        return
    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise StrictJSONError("JSON contains a non-finite number")
        return
    if isinstance(value, dict):
        members[0] += len(value)
        if members[0] > MAX_JSON_MEMBERS:
            raise StrictJSONError(f"JSON has more than {MAX_JSON_MEMBERS} container members")
        for key, item in value.items():
            if not isinstance(key, str):
                raise StrictJSONError("JSON object keys must be strings")
            _validate_string(key)
            _validate(item, depth=depth + 1, members=members)
        return
    if isinstance(value, (list, tuple)):
        members[0] += len(value)
        if members[0] > MAX_JSON_MEMBERS:
            raise StrictJSONError(f"JSON has more than {MAX_JSON_MEMBERS} container members")
        for item in value:
            _validate(item, depth=depth + 1, members=members)
        return
    raise StrictJSONError(f"value of type {type(value).__name__} is not JSON serializable")


def strict_json_loads(payload: str) -> Any:
    """Decode one bounded JSON value without Python's non-standard constants."""
    try:
        value = json.loads(
            payload,
            parse_constant=_reject_constant,
            parse_float=_parse_float,
            parse_int=_parse_int,
            object_pairs_hook=_object_from_pairs,
        )
        _validate(value, depth=0, members=[0])
    except json.JSONDecodeError as exc:
        raise StrictJSONError(f"{exc.msg} at column {exc.colno}") from exc
    except StrictJSONError:
        raise
    except (OverflowError, RecursionError, ValueError) as exc:
        raise StrictJSONError(f"JSON exceeds a safe parser bound: {exc}") from exc
    return value


def strict_json_dumps(
    value: Any,
    *,
    ensure_ascii: bool = False,
    sort_keys: bool = False,
    separators: tuple[str, str] | None = None,
) -> str:
    """Serialize finite bounded JSON or raise :class:`StrictJSONError`."""
    _validate(value, depth=0, members=[0])
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=ensure_ascii,
            sort_keys=sort_keys,
            separators=separators,
        )
    except (OverflowError, RecursionError, TypeError, ValueError) as exc:
        raise StrictJSONError(f"could not serialize strict JSON: {exc}") from exc
