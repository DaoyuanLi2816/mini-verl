"""JSONL serialization for trajectories.

One JSON object per line, validated on both write and read.  No pickle, no
``torch.save``: a trajectory file received from a stranger is data, and reading
it must never execute anything.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from pydantic import ValidationError

from miniverl.errors import SchemaValidationError
from miniverl.schemas.trajectory import (
    READABLE_TRAJECTORY_SCHEMA_VERSIONS,
    Trajectory,
)

__all__ = ["write_trajectories", "read_trajectories", "iter_trajectories", "count_trajectories"]


def write_trajectories(path: str | Path, trajectories: Iterable[Trajectory]) -> int:
    """Write trajectories as JSONL.  Returns the number written."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with p.open("w", encoding="utf-8", newline="\n") as fh:
        for traj in trajectories:
            try:
                serialized = json.dumps(
                    traj.model_dump(mode="json"),
                    ensure_ascii=False,
                    allow_nan=False,
                )
            except (OverflowError, RecursionError, TypeError, ValueError) as exc:
                raise SchemaValidationError(
                    f"trajectory {traj.trajectory_id!r} is not finite JSON: {exc}"
                ) from exc
            fh.write(serialized)
            fh.write("\n")
            written += 1
    return written


def append_trajectories(path: str | Path, trajectories: Iterable[Trajectory]) -> int:
    """Append trajectories to an existing JSONL file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with p.open("a", encoding="utf-8", newline="\n") as fh:
        for traj in trajectories:
            try:
                serialized = json.dumps(
                    traj.model_dump(mode="json"),
                    ensure_ascii=False,
                    allow_nan=False,
                )
            except (OverflowError, RecursionError, TypeError, ValueError) as exc:
                raise SchemaValidationError(
                    f"trajectory {traj.trajectory_id!r} is not finite JSON: {exc}"
                ) from exc
            fh.write(serialized)
            fh.write("\n")
            written += 1
    return written


def iter_trajectories(path: str | Path) -> Iterator[Trajectory]:
    """Stream-validate a trajectory JSONL file."""
    p = Path(path)
    if not p.is_file():
        raise SchemaValidationError(
            f"trajectory file not found: {p}",
            hint="run `miniverl demo` or `miniverl train <recipe>` first",
        )
    with p.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SchemaValidationError(f"{p}:{lineno} is not valid JSON: {exc}") from exc
            version = payload.get("schema_version")
            if version not in READABLE_TRAJECTORY_SCHEMA_VERSIONS:
                raise SchemaValidationError(
                    f"{p}:{lineno} has trajectory schema_version {version!r}, "
                    "this build reads versions "
                    f"{sorted(READABLE_TRAJECTORY_SCHEMA_VERSIONS)}"
                )
            try:
                yield Trajectory.model_validate(payload)
            except ValidationError as exc:
                raise SchemaValidationError(
                    f"{p}:{lineno} failed trajectory validation:\n{exc}"
                ) from exc


def read_trajectories(path: str | Path) -> list[Trajectory]:
    """Read and validate an entire trajectory JSONL file."""
    return list(iter_trajectories(path))


def count_trajectories(path: str | Path) -> int:
    """Count non-empty lines without fully validating them."""
    p = Path(path)
    if not p.is_file():
        return 0
    with p.open("r", encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())
