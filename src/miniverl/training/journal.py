"""Checkpoint-bound training logs; rollback tails remain available for diagnosis."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any

from miniverl.errors import CheckpointError

_LOGS = ("metrics.jsonl", "trajectories.jsonl", "rewards.jsonl", "advantages.jsonl")


def capture_logs(root: Path) -> dict[str, dict[str, Any]]:
    """Bind complete flushed log prefixes to the checkpoint state digest."""
    result = {}
    for name in _LOGS:
        path = root / name
        if path.is_symlink():
            raise CheckpointError(f"unsafe training log: {name}")
        content = path.read_bytes() if path.exists() else b""
        if content and not content.endswith(b"\n"):
            raise CheckpointError(f"cannot checkpoint an incomplete log: {name}")
        result[name] = {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
    return result


def _atomic_bytes(path: Path, content: bytes) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary).replace(path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def restore_logs(root: Path, cursors: dict[str, dict[str, Any]]) -> None:
    """Verify all prefixes before rollback; each replacement is retry-idempotent.

    The run lock excludes writers. Events are retained as an audit trail. A
    crash between replacements is safe: every file still has its bound prefix.
    """
    if not cursors:  # Historical checkpoints predate log-prefix bindings.
        return
    if set(cursors) != set(_LOGS):
        raise CheckpointError("checkpoint training-log bindings are incomplete")
    archive = root / "recovery-tails"
    journal = root / ".trajectories.jsonl.group-transaction.json"
    if archive.is_symlink() or journal.is_symlink():
        raise CheckpointError("unsafe recovery archive or trajectory journal")
    pending = []
    for name in _LOGS:
        cursor = cursors[name]
        if not isinstance(cursor, dict):
            raise CheckpointError(f"invalid checkpoint log cursor: {name}")
        size = cursor.get("bytes")
        path = root / name
        if path.is_symlink() or type(size) is not int or size < 0:
            raise CheckpointError(f"unsafe checkpoint log cursor: {name}")
        content = path.read_bytes() if path.exists() else b""
        if len(content) < size or hashlib.sha256(content[:size]).hexdigest() != cursor.get(
            "sha256"
        ):
            raise CheckpointError(f"checkpoint log prefix mismatch: {name}")
        if len(content) > size:
            pending.append((path, content[:size], content[size:]))
    if pending:
        if archive.is_symlink():
            raise CheckpointError("unsafe recovery archive directory")
        archive.mkdir(exist_ok=True)
    for path, prefix, tail in pending:
        digest = hashlib.sha256(tail).hexdigest()
        _atomic_bytes(archive / f"{path.stem}-{digest}.jsonl", tail)
        _atomic_bytes(path, prefix)
    # A killed append may have left its journal. The verified checkpoint prefix
    # supersedes that uncommitted transaction; don't apply its offsets again.
    journal = root / ".trajectories.jsonl.group-transaction.json"
    if journal.exists():
        if journal.is_symlink():
            raise CheckpointError("unsafe trajectory transaction journal")
        archive.mkdir(exist_ok=True)
        content = journal.read_bytes()
        _atomic_bytes(archive / f"transaction-{hashlib.sha256(content).hexdigest()}.json", content)
        journal.unlink()
