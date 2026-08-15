"""Transactional run-manifest publication behind the trainer facade."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Literal

from miniverl.cache.store import sha256_file
from miniverl.training.checkpoint import latest_checkpoint, validate_checkpoint
from miniverl.utils.runs import RunPaths, utc_now, write_json_atomic

__all__ = ["ManifestFinalization", "RunArtifactRecorder"]


@dataclass(frozen=True)
class ManifestFinalization:
    """Explicit state copied from the trainer at one terminal boundary."""

    status: Literal["completed", "failed", "interrupted"]
    global_step: int
    parameter_version: int
    policy_version: int
    cycles_completed: int
    rollout_policy_version: int
    projection_chunk_size: int
    chunk_size_history: tuple[int, ...]
    oom_retries: int
    final_memory: dict[str, Any]
    offline_dataset_digest: str | None
    resumed_from: dict[str, Any] | None
    require_offline_dataset: bool
    result: dict[str, Any] | None = None
    error: BaseException | None = None


class RunArtifactRecorder:
    """Own atomic manifest transitions; it never owns models or training state."""

    def __init__(self, paths: RunPaths, *, started_at: str) -> None:
        self.paths = paths
        self.started_at = started_at

    def mark_running(self) -> None:
        manifest = json.loads(self.paths.manifest.read_text(encoding="utf-8"))
        manifest["status"] = "running"
        manifest["started_at"] = self.started_at
        for key in (
            "completed_at",
            "failed_at",
            "interrupted_at",
            "closed_at",
            "failure",
        ):
            manifest.pop(key, None)
        write_json_atomic(self.paths.manifest, manifest)

    def finalize(
        self,
        finalization: ManifestFinalization,
        *,
        fallback_manifest: dict[str, Any],
    ) -> None:
        """Publish one complete terminal manifest after deriving artifact integrity."""
        if self.paths.manifest_start.is_file():
            manifest = json.loads(self.paths.manifest_start.read_text(encoding="utf-8"))
            startup_digest = hashlib.sha256(self.paths.manifest_start.read_bytes()).hexdigest()
        else:
            manifest = fallback_manifest
            startup_digest = None
        manifest.update(
            {
                "status": finalization.status,
                "started_at": self.started_at,
                "global_step": finalization.global_step,
                "global_optimizer_step": finalization.global_step,
                "parameter_version": finalization.parameter_version,
                "policy_version": finalization.policy_version,
                "rollout_iteration": finalization.cycles_completed,
                "rollout_policy_version": finalization.rollout_policy_version,
                "cycles_completed": finalization.cycles_completed,
                "actual_optimizer_updates": finalization.global_step,
                "final_projection_chunk_size": finalization.projection_chunk_size,
                "chunk_size_history": list(finalization.chunk_size_history),
                "oom_retries": finalization.oom_retries,
                "final_memory": finalization.final_memory,
                "offline_dataset": (
                    {"digest": finalization.offline_dataset_digest}
                    if finalization.offline_dataset_digest
                    else None
                ),
                "resumed_from": finalization.resumed_from,
                "startup_manifest_digest": startup_digest,
            }
        )
        for key in ("completed_at", "failed_at", "interrupted_at", "failure"):
            manifest.pop(key, None)
        now = utc_now()
        if finalization.status == "completed":
            manifest["completed_at"] = now
        elif finalization.status == "interrupted":
            manifest["interrupted_at"] = now
        else:
            manifest["failed_at"] = now
        if finalization.error is not None:
            manifest["failure"] = {
                "type": type(finalization.error).__name__,
                "message": str(finalization.error),
            }

        checkpoint_payload = None
        selected = latest_checkpoint(self.paths.checkpoints)
        if selected is not None:
            validated = validate_checkpoint(selected)
            checkpoint_payload = {
                "directory": selected.name,
                "global_step": validated.state.global_step,
                "digest": validated.content_digest,
                "integrity": validated.integrity,
            }
        manifest["final_checkpoint"] = checkpoint_payload

        eval_payload = None
        if self.paths.eval_json.is_file():
            digest, size = sha256_file(self.paths.eval_json)
            eval_payload = {
                "file": self.paths.eval_json.name,
                "digest": digest,
                "bytes": size,
            }
        manifest["final_evaluation"] = eval_payload

        required = [
            self.paths.config_original,
            self.paths.config_validated,
            self.paths.config_resolved,
            self.paths.environment,
            self.paths.manifest_start,
        ]
        if finalization.status == "completed":
            required.extend([self.paths.eval_json])
            if checkpoint_payload is None:
                required.append(self.paths.checkpoints / "final")
            if finalization.require_offline_dataset:
                required.extend(
                    [
                        self.paths.offline_dataset_manifest,
                        self.paths.offline_dataset_trajectories,
                    ]
                )
        manifest["expected_artifacts"] = {path.name: path.exists() for path in required}
        manifest["all_expected_artifacts_complete"] = (
            finalization.status == "completed"
            and all(manifest["expected_artifacts"].values())
            and checkpoint_payload is not None
        )
        if finalization.result is not None:
            manifest["result"] = finalization.result
        write_json_atomic(self.paths.manifest, manifest)
