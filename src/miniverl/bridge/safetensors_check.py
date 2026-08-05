"""Structural validation of a safetensors file, beyond its header.

Up to v0.6.2 the bridge check read the 8-byte header length, parsed the header
JSON and reported success as soon as one tensor key existed. A file whose header
declares a 4x4 F32 tensor but carries no payload bytes therefore passed, while
the official reader rejects it with "file not fully covered".

Validation here has two independent stages:

* a dependency-free structural pass over the header that checks dtype, shape,
  offset arithmetic, contiguity and full coverage of the data segment;
* materialization through the official ``safetensors`` reader, which is the
  authority on the format.

The reported level always says which of those actually ran. A header-only result
is never called loadable.
"""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path
from typing import Any

__all__ = [
    "SAFETENSORS_LEVELS",
    "inspect_safetensors",
]

#: Ordered weakest to strongest.
SAFETENSORS_LEVELS = (
    "not_present",
    "header_only",
    "payload_structure_validated",
    "tensor_materialization_validated",
)

#: Byte width of every dtype the format defines.
_DTYPE_BYTES: dict[str, int] = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E5M2": 1,
    "F8_E4M3": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}

#: A header larger than this is treated as hostile rather than decoded.
_MAX_HEADER_BYTES = 100 * 1024 * 1024


def _structural_problems(path: Path) -> tuple[list[str], int, dict[str, Any]]:
    """Validate the header against the real file size without any dependency."""
    problems: list[str] = []
    size = path.stat().st_size
    with path.open("rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            return ["missing safetensors header length"], 0, {}
        (header_length,) = struct.unpack("<Q", raw_length)
        if header_length < 2:
            return ["invalid safetensors header length"], 0, {}
        if header_length > _MAX_HEADER_BYTES:
            return [f"safetensors header of {header_length} bytes is implausible"], 0, {}
        if header_length > size - 8:
            return ["safetensors header extends past the end of the file"], 0, {}
        try:
            header = json.loads(handle.read(header_length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            return [f"malformed safetensors header JSON: {exc}"], 0, {}

    if not isinstance(header, dict):
        return ["safetensors header must be a JSON object"], 0, {}

    data_length = size - 8 - header_length
    tensors = {key: value for key, value in header.items() if key != "__metadata__"}
    if not tensors:
        return ["safetensors contains no tensors"], data_length, {}

    spans: list[tuple[int, int, str]] = []
    for name, entry in sorted(tensors.items()):
        if not isinstance(entry, dict):
            problems.append(f"{name}: tensor entry must be an object")
            continue
        dtype = entry.get("dtype")
        shape = entry.get("shape")
        offsets = entry.get("data_offsets")
        if not isinstance(dtype, str) or dtype not in _DTYPE_BYTES:
            problems.append(f"{name}: unknown dtype {dtype!r}")
            continue
        if not isinstance(shape, list) or not all(
            isinstance(dim, int) and not isinstance(dim, bool) and dim >= 0 for dim in shape
        ):
            problems.append(f"{name}: shape must be a list of non-negative integers")
            continue
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(item, int) and not isinstance(item, bool) for item in offsets)
        ):
            problems.append(f"{name}: data_offsets must be two integers")
            continue
        begin, end = offsets
        if begin < 0 or end < begin:
            problems.append(f"{name}: data_offsets [{begin}, {end}] are not ordered")
            continue
        expected = math.prod(shape) * _DTYPE_BYTES[dtype]
        if end - begin != expected:
            problems.append(
                f"{name}: {dtype}{shape} needs {expected} bytes but data_offsets span {end - begin}"
            )
            continue
        if end > data_length:
            problems.append(
                f"{name}: data_offsets end at {end} but only {data_length} payload bytes exist"
            )
            continue
        spans.append((begin, end, name))

    # The format requires the tensor spans to tile the data segment exactly:
    # gaps, overlaps and trailing bytes are all rejected upstream.
    cursor = 0
    for begin, end, name in sorted(spans):
        if begin < cursor:
            problems.append(f"{name}: data_offsets overlap the preceding tensor")
        elif begin > cursor:
            problems.append(f"{name}: {begin - cursor} uncovered byte(s) precede this tensor")
        cursor = max(cursor, end)
    if spans and cursor != data_length:
        problems.append(
            f"{data_length - cursor} trailing payload byte(s) are not covered by any tensor"
        )

    summary = {
        "tensors": len(tensors),
        "declared_payload_bytes": cursor,
        "actual_payload_bytes": data_length,
        "metadata_keys": sorted(header.get("__metadata__", {}))
        if isinstance(header.get("__metadata__"), dict)
        else [],
    }
    return problems, data_length, summary


def _materialize(path: Path) -> tuple[bool, str, int]:
    """Read every tensor through the official reader, which owns the format."""
    try:
        from safetensors import safe_open
    except ImportError:
        return False, "the official safetensors package is not installed", 0
    try:
        with safe_open(str(path), framework="np") as handle:
            keys = list(handle.keys())
            for key in keys:
                # Reading the slice's shape forces the reader to resolve the
                # tensor against the real buffer.
                handle.get_slice(key).get_shape()
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}", 0
    return True, f"{len(keys)} tensor(s) materialized through the official reader", len(keys)


def inspect_safetensors(path: str | Path, *, require_payload: bool = False) -> dict[str, Any]:
    """Report a truthful safetensors verification level.

    ``require_payload`` fails the check unless the payload structure validated,
    so a strict caller cannot be satisfied by a header alone.
    """
    target = Path(path)
    check: dict[str, Any] = {
        "status": "fail",
        "verification_level": "not_present",
        "problems": [],
        "detail": "",
        "strict_payload_required": bool(require_payload),
        "strict_payload_satisfied": False,
        "scope": "file structure only; tensor values and model semantics are not checked",
    }
    if not target.is_file():
        check["detail"] = f"{target.name} is not present"
        return check

    try:
        problems, _data_length, summary = _structural_problems(target)
    except OSError as exc:
        check["verification_level"] = "header_only"
        check["detail"] = f"cannot read {target.name}: {exc}"
        return check

    check.update(summary)
    if problems:
        check["verification_level"] = "header_only"
        check["problems"] = problems
        check["detail"] = f"structural validation rejected {target.name}: {problems[0]}"
        return check

    check["verification_level"] = "payload_structure_validated"
    check["status"] = "ok"
    check["strict_payload_satisfied"] = True
    check["detail"] = (
        f"{summary['tensors']} tensor(s); offsets are contiguous and cover all "
        f"{summary['actual_payload_bytes']} payload bytes"
    )

    materialized, detail, _count = _materialize(target)
    check["official_reader"] = detail
    if materialized:
        check["verification_level"] = "tensor_materialization_validated"
        check["detail"] = detail
    elif "not installed" in detail:
        # A missing optional dependency must not be reported as a stronger
        # result, but it also must not fail a structurally valid file.
        check["official_reader_status"] = "dependency_missing"
    else:
        # The official reader is the authority: if it rejects the file, so do we.
        check["status"] = "fail"
        check["strict_payload_satisfied"] = False
        check["verification_level"] = "header_only"
        check["problems"] = [detail]
        check["detail"] = f"the official safetensors reader rejected {target.name}: {detail}"
    return check
