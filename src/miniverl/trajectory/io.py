"""JSONL serialization for trajectories.

One JSON object per line, validated on both write and read.  No pickle, no
``torch.save``: a trajectory file received from a stranger is data, and reading
it must never execute anything.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Iterator
from pathlib import Path

from pydantic import ValidationError

from miniverl.errors import SchemaValidationError
from miniverl.schemas.trajectory import (
    READABLE_TRAJECTORY_SCHEMA_VERSIONS,
    TRAJECTORY_SCHEMA_VERSION,
    Trajectory,
    derive_grouped_trajectory_id,
    validate_trajectory_groups,
)
from miniverl.utils.runs import write_json_atomic

__all__ = [
    "append_trajectory_groups",
    "append_trajectories",
    "count_trajectories",
    "iter_trajectories",
    "read_trajectories",
    "write_trajectories",
]


def _serialize_trajectory(traj: Trajectory) -> str:
    try:
        return json.dumps(
            traj.model_dump(mode="json"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (OverflowError, RecursionError, TypeError, ValueError) as exc:
        raise SchemaValidationError(
            f"trajectory {traj.trajectory_id!r} is not finite JSON: {exc}"
        ) from exc


def _upgrade_for_write(traj: Trajectory) -> Trajectory:
    """Upgrade an in-memory v1/v2 trajectory to the canonical n=1 v3 form."""

    if traj.schema_version == TRAJECTORY_SCHEMA_VERSION:
        return traj
    legacy_id = traj.trajectory_id
    prompt_payload = {
        "task_id": traj.task_id,
        "environment": traj.environment,
        "prompt_token_ids": [
            token_id
            for token_id, generated in zip(
                traj.token_ids,
                traj.model_generated_mask,
                strict=True,
            )
            if not generated
        ],
    }
    prompt_digest = hashlib.sha256(
        json.dumps(prompt_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    policy_payload = {
        "model_id": traj.model_id,
        "model_revision": traj.model_revision,
        "policy_version": traj.policy_version,
        "tokenizer_fingerprint": traj.tokenizer_fingerprint,
    }
    policy_digest = hashlib.sha256(
        json.dumps(policy_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    raw_seed = traj.metadata.get("generation_seed", traj.metadata.get("seed", 0))
    generation_seed = raw_seed if isinstance(raw_seed, int) and raw_seed >= 0 else 0
    group_digest = hashlib.sha256(
        f"{traj.environment}\0{traj.task_id}\0{legacy_id}\0{prompt_digest}".encode()
    ).hexdigest()
    prompt_group_id = f"n1-{group_digest[:24]}"
    rollout_backend = traj.metadata.get("rollout_backend", "environment_direct")
    if not isinstance(rollout_backend, str) or not rollout_backend:
        rollout_backend = "environment_direct"
    metadata = dict(traj.metadata)
    metadata.setdefault("legacy_trajectory_id", legacy_id)
    payload = traj.model_dump(mode="json")
    payload.update(
        {
            "schema_version": TRAJECTORY_SCHEMA_VERSION,
            "trajectory_id": derive_grouped_trajectory_id(
                prompt_group_id=prompt_group_id,
                sample_index=0,
                rollout_policy_identity_digest=policy_digest,
                generation_seed=generation_seed,
            ),
            "prompt_group_id": prompt_group_id,
            "prompt_digest": prompt_digest,
            "sample_index": 0,
            "samples_per_prompt": 1,
            "generation_seed": generation_seed,
            "rollout_backend": rollout_backend,
            "rollout_policy_identity_digest": policy_digest,
            "metadata": metadata,
        }
    )
    upgraded = Trajectory.model_validate(payload)
    for field_name in Trajectory.model_fields:
        setattr(traj, field_name, getattr(upgraded, field_name))
    return traj


def write_trajectories(path: str | Path, trajectories: Iterable[Trajectory]) -> int:
    """Write trajectories as JSONL.  Returns the number written."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with p.open("w", encoding="utf-8", newline="\n") as fh:
        for traj in trajectories:
            traj = _upgrade_for_write(traj)
            serialized = _serialize_trajectory(traj)
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
            traj = _upgrade_for_write(traj)
            serialized = _serialize_trajectory(traj)
            fh.write(serialized)
            fh.write("\n")
            written += 1
    return written


def _group_journal(path: Path) -> Path:
    return path.with_name(f".{path.name}.group-transaction.json")


def _recover_group_append(path: Path) -> None:
    journal = _group_journal(path)
    if not journal.is_file():
        return
    try:
        payload = json.loads(journal.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError) as exc:
        raise SchemaValidationError(f"cannot recover trajectory group journal: {exc}") from exc
    status = payload.get("status")
    if status == "pending":
        start_offset = payload.get("start_offset")
        if not isinstance(start_offset, int) or start_offset < 0:
            raise SchemaValidationError("trajectory group journal has an invalid start_offset")
        if path.exists():
            with path.open("r+b") as handle:
                if handle.seek(0, os.SEEK_END) < start_offset:
                    raise SchemaValidationError(
                        "trajectory file is shorter than its pending group transaction"
                    )
                handle.truncate(start_offset)
                handle.flush()
                os.fsync(handle.fileno())
    elif status != "committed":
        raise SchemaValidationError("trajectory group journal has an unknown status")
    journal.unlink()


def append_trajectory_groups(
    path: str | Path,
    trajectories: Iterable[Trajectory],
    *,
    transaction_id: str,
) -> int:
    """Atomically append one or more complete schema-v3 prompt groups.

    The sibling journal is published before data bytes. A process crash before
    the committed journal is published is repaired by truncating back to the
    recorded byte offset on the next call. All logical groups supplied here
    therefore become visible together or not at all.
    """

    if not transaction_id:
        raise ValueError("transaction_id cannot be empty")
    rows = list(trajectories)
    validate_trajectory_groups(rows)
    if any(row.schema_version != TRAJECTORY_SCHEMA_VERSION for row in rows):
        raise ValueError("transactional group writer only publishes the current schema")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    _recover_group_append(p)
    requested = {row.trajectory_id: row for row in rows}
    existing: dict[str, Trajectory] = {}
    if p.is_file():
        for row in iter_trajectories(p):
            if row.trajectory_id not in requested:
                continue
            if row.trajectory_id in existing:
                raise SchemaValidationError(
                    f"trajectory file already contains duplicate id {row.trajectory_id!r}"
                )
            existing[row.trajectory_id] = row
    if existing:
        if set(existing) != set(requested):
            raise ValueError(
                "transactional trajectory group is partially present; refusing to append "
                "a mixed committed/uncommitted replay"
            )
        mismatched = [
            trajectory_id
            for trajectory_id, row in requested.items()
            if existing[trajectory_id].model_dump(mode="json") != row.model_dump(mode="json")
        ]
        if mismatched:
            raise SchemaValidationError(
                "committed trajectory replay changed payload for " + ", ".join(sorted(mismatched))
            )
        return 0
    encoded = "".join(f"{_serialize_trajectory(row)}\n" for row in rows).encode("utf-8")
    start_offset = p.stat().st_size if p.exists() else 0
    journal = _group_journal(p)
    groups = sorted({str(row.prompt_group_id) for row in rows})
    write_json_atomic(
        journal,
        {
            "schema_version": 1,
            "status": "pending",
            "transaction_id": transaction_id,
            "start_offset": start_offset,
            "groups": groups,
            "trajectories": len(rows),
        },
    )
    try:
        with p.open("ab") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        end_offset = start_offset + len(encoded)
        write_json_atomic(
            journal,
            {
                "schema_version": 1,
                "status": "committed",
                "transaction_id": transaction_id,
                "start_offset": start_offset,
                "end_offset": end_offset,
                "groups": groups,
                "trajectories": len(rows),
            },
        )
    except BaseException:
        if p.exists():
            with p.open("r+b") as handle:
                handle.truncate(start_offset)
                handle.flush()
                os.fsync(handle.fileno())
        journal.unlink(missing_ok=True)
        raise
    journal.unlink()
    return len(rows)


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
