"""Content-addressed fixed dataset artifacts for exact offline-KD resume."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

from miniverl.cache.store import TeacherCache, sha256_file
from miniverl.errors import CheckpointError
from miniverl.trajectory.io import read_trajectories, write_trajectories
from miniverl.utils.runs import canonical_json, utc_now, write_json

if TYPE_CHECKING:
    from miniverl.config.models import RunConfig
    from miniverl.schemas.trajectory import Trajectory
    from miniverl.training.trainer import TrainSample
    from miniverl.utils.runs import RunPaths

__all__ = ["create_offline_dataset", "load_offline_dataset"]

OFFLINE_DATASET_SCHEMA_VERSION = 1


def _digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _stable_identity(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": payload["schema_version"],
        "trajectory_ids": payload["trajectory_ids"],
        "task_ids": payload["task_ids"],
        "split": payload["split"],
        "seed": payload["seed"],
        "tokenizer_identity": payload["tokenizer_identity"],
        "policy_version": payload["policy_version"],
        "teacher_identity": payload["teacher_identity"],
        "trajectory_sha256": payload["files"]["trajectories.jsonl"]["sha256"],
        "cache_entry_checksums": payload["cache_entry_checksums"],
        "cache_shard_checksums": payload["cache_shard_checksums"],
        "selected_positions": payload["selected_positions"],
        "creation_policy": payload["creation_policy"],
        "source": payload["source"],
    }


def create_offline_dataset(
    paths: RunPaths,
    *,
    samples: list[TrainSample],
    cache: TeacherCache,
    config: RunConfig,
    tokenizer_identity: dict[str, Any],
    teacher_identity: dict[str, Any],
) -> str:
    """Persist the fixed ordered trajectories and a checksummed manifest once."""
    root = paths.offline_dataset
    if paths.offline_dataset_manifest.exists():
        raise CheckpointError(
            f"offline dataset already exists at {root}",
            hint="resume and validate the existing dataset instead of regenerating it",
        )
    root.mkdir(parents=True, exist_ok=False)
    trajectories = [sample.trajectory for sample in samples]
    write_trajectories(paths.offline_dataset_trajectories, trajectories)
    trajectory_sha, trajectory_size = sha256_file(paths.offline_dataset_trajectories)
    cache_index = cache.path / "index.json"
    cache_sha, cache_size = sha256_file(cache_index)
    payload: dict[str, Any] = {
        "schema_version": OFFLINE_DATASET_SCHEMA_VERSION,
        "created_at": utc_now(),
        "trajectory_ids": [trajectory.trajectory_id for trajectory in trajectories],
        "task_ids": [trajectory.task_id for trajectory in trajectories],
        "split": "train",
        "seed": config.run.seed,
        "tokenizer_identity": tokenizer_identity,
        "policy_version": trajectories[0].policy_version if trajectories else 0,
        "teacher_identity": teacher_identity,
        "cache_index_digest": cache_sha,
        "cache_entry_checksums": {
            trajectory_id: entry.checksum
            for trajectory_id, entry in sorted(cache.index.entries.items())
        },
        "cache_shard_checksums": {
            name: shard.sha256 for name, shard in sorted(cache.index.shards.items())
        },
        "files": {
            "trajectories.jsonl": {
                "sha256": trajectory_sha,
                "bytes": trajectory_size,
            },
            "teacher-cache/index.json": {
                "sha256": cache_sha,
                "bytes": cache_size,
            },
        },
        "selected_positions": [
            {
                "trajectory_id": sample.trajectory.trajectory_id,
                "count": len(sample.alignment.student_prediction_positions),
                "positions": list(sample.alignment.student_prediction_positions),
            }
            for sample in samples
        ],
        "creation_policy": "create_once_then_validate",
        "source": "environment_oracle",
    }
    payload["dataset_digest"] = _digest(_stable_identity(payload))
    write_json(paths.offline_dataset_manifest, payload)
    return str(payload["dataset_digest"])


def load_offline_dataset(
    paths: RunPaths,
    *,
    cache: TeacherCache,
    expected_digest: str,
) -> tuple[dict[str, Any], list[Trajectory]]:
    """Validate every persisted file and return trajectories in manifest order."""
    manifest_path = paths.offline_dataset_manifest
    if not manifest_path.is_file():
        raise CheckpointError(
            f"offline-KD checkpoint requires the persisted dataset at {manifest_path}"
        )
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CheckpointError(f"cannot read offline dataset manifest: {exc}") from exc
    if payload.get("schema_version") != OFFLINE_DATASET_SCHEMA_VERSION:
        raise CheckpointError(
            f"offline dataset schema_version {payload.get('schema_version')!r} is not readable"
        )
    actual_digest = _digest(_stable_identity(payload))
    recorded_digest = payload.get("dataset_digest")
    if actual_digest != recorded_digest or (expected_digest and actual_digest != expected_digest):
        raise CheckpointError("offline dataset digest does not match its checkpoint")

    files = payload.get("files", {})
    for name, path in (
        ("trajectories.jsonl", paths.offline_dataset_trajectories),
        ("teacher-cache/index.json", cache.path / "index.json"),
    ):
        expected = files.get(name, {})
        actual_sha, actual_size = sha256_file(path)
        if actual_sha != expected.get("sha256") or actual_size != expected.get("bytes"):
            raise CheckpointError(f"offline dataset file checksum mismatch: {name}")
    if payload.get("cache_index_digest") != files["teacher-cache/index.json"]["sha256"]:
        raise CheckpointError("offline dataset cache index digest is inconsistent")
    if payload.get("cache_entry_checksums") != {
        trajectory_id: entry.checksum
        for trajectory_id, entry in sorted(cache.index.entries.items())
    }:
        raise CheckpointError("offline dataset cache entry checksums changed")
    if payload.get("cache_shard_checksums") != {
        name: shard.sha256 for name, shard in sorted(cache.index.shards.items())
    }:
        raise CheckpointError("offline dataset cache shard checksums changed")

    trajectories = read_trajectories(paths.offline_dataset_trajectories)
    trajectory_ids = [trajectory.trajectory_id for trajectory in trajectories]
    if trajectory_ids != payload.get("trajectory_ids"):
        raise CheckpointError("offline dataset trajectory order does not match its manifest")
    if [trajectory.task_id for trajectory in trajectories] != payload.get("task_ids"):
        raise CheckpointError("offline dataset task order does not match its manifest")
    return payload, trajectories
