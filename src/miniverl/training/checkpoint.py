"""Pickle-free, checksummed and atomic checkpointing.

A complete checkpoint contains trainable weights, JSON training state and a
manifest written last. Optimizer tensors are optional because an optimizer may
not have accumulated moments yet. New checkpoints are checksummed; v0.2
checkpoints without a manifest remain readable and are explicitly reported as
``legacy_unchecksummed``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from miniverl.errors import CheckpointError
from miniverl.utils.runs import canonical_json, utc_now, write_text
from miniverl.utils.seeding import RngSnapshot

__all__ = [
    "CheckpointState",
    "ValidatedCheckpoint",
    "latest_checkpoint",
    "list_checkpoints",
    "load_checkpoint",
    "save_checkpoint",
    "validate_checkpoint",
]

_ADAPTER = "adapter.safetensors"
_OPTIMIZER = "optimizer.safetensors"
_STATE = "state.json"
_MANIFEST = "checkpoint.json"
CHECKPOINT_SCHEMA_VERSION = 1
CHECKPOINT_MANIFEST_SCHEMA_VERSION = 1


@dataclass
class CheckpointState:
    """Non-tensor training state."""

    schema_version: int = CHECKPOINT_SCHEMA_VERSION
    miniverl_version: str = ""
    global_step: int = 0
    policy_version: int = 0
    # Explicit name used by v0.2.1+. ``policy_version`` remains the
    # backward-compatible serialized alias.
    parameter_version: int | None = None
    cycle: int = 0
    rollout_iteration: int | None = None
    rollout_policy_version: int | None = None
    task_cursor: int = 0
    scheduler: dict[str, Any] = field(default_factory=dict)
    scaler: dict[str, Any] | None = None
    optimizer_param_groups: list[dict[str, Any]] = field(default_factory=list)
    optimizer_state_keys: list[str] = field(default_factory=list)
    optimizer_scalars: dict[str, Any] = field(default_factory=dict)
    rng: dict[str, Any] = field(default_factory=dict)
    config_digest: str = ""
    resolved_config_digest: str = ""
    offline_dataset_digest: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly view."""
        return {
            "schema_version": self.schema_version,
            "miniverl_version": self.miniverl_version,
            "global_step": self.global_step,
            "policy_version": self.policy_version,
            "parameter_version": (
                self.policy_version if self.parameter_version is None else self.parameter_version
            ),
            "cycle": self.cycle,
            "rollout_iteration": (
                self.cycle if self.rollout_iteration is None else self.rollout_iteration
            ),
            "rollout_policy_version": (
                self.policy_version
                if self.rollout_policy_version is None
                else self.rollout_policy_version
            ),
            "task_cursor": self.task_cursor,
            "scheduler": self.scheduler,
            "scaler": self.scaler,
            "optimizer_param_groups": self.optimizer_param_groups,
            "optimizer_state_keys": self.optimizer_state_keys,
            "optimizer_scalars": self.optimizer_scalars,
            "rng": self.rng,
            "config_digest": self.config_digest,
            "resolved_config_digest": self.resolved_config_digest,
            "offline_dataset_digest": self.offline_dataset_digest,
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CheckpointState:
        """Rebuild from :meth:`to_dict` output, including v0.2 state files."""
        version = payload.get("schema_version")
        if version != CHECKPOINT_SCHEMA_VERSION:
            raise CheckpointError(
                f"checkpoint schema_version {version!r} is not readable by this build "
                f"(expected {CHECKPOINT_SCHEMA_VERSION})"
            )
        fields = cls.__dataclass_fields__
        unknown = sorted(set(payload).difference(fields))
        if unknown:
            raise CheckpointError(f"checkpoint state has unknown fields: {', '.join(unknown)}")
        return cls(**payload)


@dataclass(frozen=True)
class ValidatedCheckpoint:
    """Validated checkpoint metadata, before any model state is mutated."""

    path: Path
    state: CheckpointState
    integrity: str
    content_digest: str
    identity: dict[str, Any] = field(default_factory=dict)
    manifest: dict[str, Any] | None = None


def _split_optimizer_state(
    optimizer: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], list[str]]:
    """Separate an optimizer state dict into tensors, groups and scalars."""
    import torch

    if optimizer is None:
        return {}, [], {}, []
    state = optimizer.state_dict()
    tensors: dict[str, Any] = {}
    scalars: dict[str, Any] = {}
    keys: list[str] = []
    for param_id, entry in state.get("state", {}).items():
        for name, value in entry.items():
            key = f"{param_id}|{name}"
            keys.append(key)
            if isinstance(value, torch.Tensor):
                tensors[key] = value.detach().to("cpu").contiguous()
            else:
                scalars[key] = value
    groups = [dict(group) for group in state.get("param_groups", [])]
    return tensors, groups, scalars, keys


def _rebuild_optimizer_state(
    tensors: dict[str, Any],
    scalars: dict[str, Any],
    keys: list[str],
    groups: list[dict[str, Any]],
) -> dict[str, Any]:
    groups = [dict(group) for group in groups]
    for group in groups:
        if isinstance(group.get("betas"), list):
            group["betas"] = tuple(group["betas"])
    state: dict[int, dict[str, Any]] = {}
    for key in keys:
        param_id_text, name = key.split("|", 1)
        param_id = int(param_id_text)
        entry = state.setdefault(param_id, {})
        if key in tensors:
            entry[name] = tensors[key]
        elif key in scalars:
            entry[name] = scalars[key]
        else:
            raise CheckpointError(f"optimizer state key {key!r} is missing from the checkpoint")
    return {"state": state, "param_groups": groups}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def _content_digest(
    *,
    state: CheckpointState,
    files: dict[str, dict[str, Any]],
    identity: dict[str, Any],
) -> str:
    payload = {
        "state": state.to_dict(),
        "files": files,
        "identity": identity,
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _sync_file(path: Path) -> None:
    try:
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
    except OSError:
        # Some Windows/OneDrive filesystem combinations reject fsync on a
        # read-only descriptor. The manifest-last + sibling-directory swap
        # still prevents a partial checkpoint from becoming discoverable.
        if os.name != "nt":
            raise


def _replace_directory_atomically(source: Path, target: Path) -> None:
    """Swap a complete sibling directory into place, restoring on failure."""
    backup: Path | None = None
    if target.exists():
        if not target.is_dir() or target.is_symlink():
            raise CheckpointError(f"checkpoint target is not a safe directory: {target}")
        backup = target.with_name(f".{target.name}.backup-{uuid.uuid4().hex}")
        target.replace(backup)
    try:
        source.replace(target)
    except Exception:
        if backup is not None and backup.exists() and not target.exists():
            backup.replace(target)
        raise
    if backup is not None:
        shutil.rmtree(backup)


def save_checkpoint(
    directory: str | Path,
    *,
    trainable_state: dict[str, Any],
    optimizer: Any,
    state: CheckpointState,
    rng: RngSnapshot,
    identity: dict[str, Any] | None = None,
) -> Path:
    """Write a complete checkpoint to a sibling temporary directory, then swap."""
    from safetensors.torch import save_file

    target = Path(directory)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{uuid.uuid4().hex}")
    temporary.mkdir()
    checkpoint_identity = dict(identity or {})
    try:
        if not trainable_state:
            raise CheckpointError(
                "cannot save a checkpoint without trainable weights",
                hint="verify that the student exposes a non-empty trainable_state_dict",
            )
        save_file(
            {key: value.detach().to("cpu").contiguous() for key, value in trainable_state.items()},
            str(temporary / _ADAPTER),
            metadata={"miniverl": "adapter"},
        )
        tensors, groups, scalars, keys = _split_optimizer_state(optimizer)
        if tensors:
            save_file(
                tensors,
                str(temporary / _OPTIMIZER),
                metadata={"miniverl": "optimizer"},
            )
        state.optimizer_param_groups = groups
        state.optimizer_scalars = {key: _jsonable(value) for key, value in scalars.items()}
        state.optimizer_state_keys = keys
        state.rng = rng.to_dict()
        try:
            state_json = (
                json.dumps(
                    state.to_dict(),
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                    default=_jsonable,
                )
                + "\n"
            )
        except (OverflowError, RecursionError, TypeError, ValueError) as exc:
            raise CheckpointError(f"checkpoint state is not finite JSON: {exc}") from exc
        write_text(temporary / _STATE, state_json)

        files = {
            path.name: _file_record(path) for path in sorted(temporary.iterdir()) if path.is_file()
        }
        manifest = {
            "schema_version": CHECKPOINT_MANIFEST_SCHEMA_VERSION,
            "complete": True,
            "created_at": utc_now(),
            "miniverl_version": state.miniverl_version,
            "global_step": state.global_step,
            "policy_version": state.policy_version,
            "parameter_version": (
                state.policy_version if state.parameter_version is None else state.parameter_version
            ),
            "rollout_iteration": (
                state.cycle if state.rollout_iteration is None else state.rollout_iteration
            ),
            "rollout_policy_version": (
                state.policy_version
                if state.rollout_policy_version is None
                else state.rollout_policy_version
            ),
            "config_digest": state.config_digest,
            "resolved_config_digest": state.resolved_config_digest,
            "offline_dataset_digest": state.offline_dataset_digest,
            "identity": checkpoint_identity,
            "files": files,
            "content_digest": _content_digest(
                state=state,
                files=files,
                identity=checkpoint_identity,
            ),
        }
        try:
            manifest_json = json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
        except (OverflowError, RecursionError, TypeError, ValueError) as exc:
            raise CheckpointError(f"checkpoint manifest is not finite JSON: {exc}") from exc
        write_text(temporary / _MANIFEST, manifest_json)
        for path in temporary.iterdir():
            if path.is_file():
                _sync_file(path)
        validate_checkpoint(temporary)
        _replace_directory_atomically(temporary, target)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return target


def validate_checkpoint(directory: str | Path) -> ValidatedCheckpoint:
    """Validate completeness and checksums without mutating a backend."""
    target = Path(directory)
    if not target.is_dir():
        raise CheckpointError(f"checkpoint directory not found: {target}")
    state_path = target / _STATE
    if not state_path.is_file():
        raise CheckpointError(
            f"{target} is not a miniVERL checkpoint (missing {_STATE})",
            hint="pass a directory such as runs/<run-id>/checkpoints/step-000010",
        )
    try:
        state_payload = json.loads(state_path.read_text(encoding="utf-8"))
        state = CheckpointState.from_dict(state_payload)
    except CheckpointError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError) as exc:
        raise CheckpointError(f"cannot read checkpoint state {state_path}: {exc}") from exc

    adapter_path = target / _ADAPTER
    if not adapter_path.is_file():
        raise CheckpointError(f"checkpoint is incomplete (missing {_ADAPTER}): {target}")

    manifest_path = target / _MANIFEST
    if not manifest_path.is_file():
        files = {
            path.name: _file_record(path) for path in sorted(target.iterdir()) if path.is_file()
        }
        return ValidatedCheckpoint(
            path=target,
            state=state,
            integrity="legacy_unchecksummed",
            content_digest=_content_digest(state=state, files=files, identity={}),
        )

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError) as exc:
        raise CheckpointError(f"cannot read checkpoint manifest {manifest_path}: {exc}") from exc
    if manifest.get("schema_version") != CHECKPOINT_MANIFEST_SCHEMA_VERSION:
        raise CheckpointError(
            f"checkpoint manifest schema_version {manifest.get('schema_version')!r} "
            f"is not readable (expected {CHECKPOINT_MANIFEST_SCHEMA_VERSION})"
        )
    if manifest.get("complete") is not True:
        raise CheckpointError(f"checkpoint manifest is not marked complete: {target}")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise CheckpointError(f"checkpoint manifest has no valid files table: {target}")
    for required in (_ADAPTER, _STATE):
        if required not in files:
            raise CheckpointError(
                f"checkpoint manifest is incomplete (missing {required}): {target}"
            )
    for name, expected in files.items():
        path = target / name
        if not isinstance(name, str) or Path(name).name != name or not isinstance(expected, dict):
            raise CheckpointError(f"checkpoint manifest contains an unsafe file entry: {name!r}")
        if not path.is_file():
            raise CheckpointError(f"checkpoint file listed in manifest is missing: {name}")
        actual_bytes = path.stat().st_size
        expected_bytes = expected.get("bytes")
        if actual_bytes != expected_bytes:
            raise CheckpointError(
                f"checkpoint checksum validation failed for {name}: "
                f"size {actual_bytes} != {expected_bytes}"
            )
        actual_sha256 = _sha256_file(path)
        if actual_sha256 != expected.get("sha256"):
            raise CheckpointError(f"checkpoint checksum validation failed for {name}")
    if manifest.get("global_step") != state.global_step:
        raise CheckpointError("checkpoint manifest and state disagree on global_step")
    if manifest.get("policy_version") != state.policy_version:
        raise CheckpointError("checkpoint manifest and state disagree on policy_version")
    parameter_version = (
        state.policy_version if state.parameter_version is None else state.parameter_version
    )
    if manifest.get("parameter_version", parameter_version) != parameter_version:
        raise CheckpointError("checkpoint manifest and state disagree on parameter_version")
    rollout_iteration = state.cycle if state.rollout_iteration is None else state.rollout_iteration
    if manifest.get("rollout_iteration", rollout_iteration) != rollout_iteration:
        raise CheckpointError("checkpoint manifest and state disagree on rollout_iteration")
    rollout_policy_version = (
        state.policy_version
        if state.rollout_policy_version is None
        else state.rollout_policy_version
    )
    if manifest.get("rollout_policy_version", rollout_policy_version) != rollout_policy_version:
        raise CheckpointError("checkpoint manifest and state disagree on rollout_policy_version")
    identity = manifest.get("identity", {})
    if not isinstance(identity, dict):
        raise CheckpointError("checkpoint manifest identity must be an object")
    actual_digest = _content_digest(state=state, files=files, identity=identity)
    if actual_digest != manifest.get("content_digest"):
        raise CheckpointError("checkpoint manifest content checksum is inconsistent")
    return ValidatedCheckpoint(
        path=target,
        state=state,
        integrity="checksummed_v1",
        content_digest=actual_digest,
        identity=identity,
        manifest=manifest,
    )


def load_checkpoint(
    directory: str | Path,
    *,
    backend: Any,
    optimizer: Any,
    device: str = "cpu",
    include_optimizer: bool = True,
    include_rng: bool = True,
    expected_config_digest: str | None = None,
    expected_identity: dict[str, Any] | None = None,
) -> CheckpointState:
    """Validate fully, then restore weights and optional optimizer/RNG state."""
    from safetensors.torch import load_file

    from miniverl.utils.seeding import restore_rng

    validated = validate_checkpoint(directory)
    state = validated.state
    if expected_config_digest and state.config_digest != expected_config_digest:
        raise CheckpointError(
            "the checkpoint was written by a different configuration",
            hint="resume with the original config.resolved.yaml",
        )
    if expected_identity:
        mismatches = {
            key: (validated.identity.get(key), value)
            for key, value in expected_identity.items()
            if validated.identity.get(key) != value
        }
        if mismatches:
            details = ", ".join(
                f"{key}: checkpoint={actual!r}, current={expected!r}"
                for key, (actual, expected) in sorted(mismatches.items())
            )
            raise CheckpointError(f"checkpoint identity mismatch ({details})")

    target = validated.path
    adapter = load_file(str(target / _ADAPTER), device=device)
    optimizer_state: dict[str, Any] | None = None
    if include_optimizer:
        if optimizer is None:
            raise CheckpointError(
                "optimizer restoration was requested but no optimizer was provided"
            )
        optimizer_path = target / _OPTIMIZER
        tensors = load_file(str(optimizer_path), device=device) if optimizer_path.is_file() else {}
        if state.optimizer_state_keys:
            optimizer_state = _rebuild_optimizer_state(
                tensors,
                state.optimizer_scalars,
                state.optimizer_state_keys,
                state.optimizer_param_groups,
            )

    backend.load_trainable_state_dict(adapter)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    if include_rng and state.rng:
        restore_rng(RngSnapshot.from_dict(state.rng))
    return state


def list_checkpoints(directory: str | Path) -> list[Path]:
    """Return valid checkpoint directories sorted by state step, then name."""
    root = Path(directory)
    if not root.is_dir():
        return []
    validated: list[ValidatedCheckpoint] = []
    for path in root.iterdir():
        if not path.is_dir() or path.name.startswith(".") or ".tmp-" in path.name:
            continue
        if not (path / _STATE).is_file():
            continue
        validated.append(validate_checkpoint(path))
    return [
        item.path
        for item in sorted(validated, key=lambda item: (item.state.global_step, item.path.name))
    ]


def latest_checkpoint(directory: str | Path) -> Path | None:
    """Select the unique highest state step, preferring ``final`` for duplicates."""
    root = Path(directory)
    if not root.is_dir():
        return None
    candidates: list[ValidatedCheckpoint] = []
    for path in root.iterdir():
        if not path.is_dir() or path.name.startswith(".") or ".tmp-" in path.name:
            continue
        if not (path / _STATE).is_file():
            continue
        candidates.append(validate_checkpoint(path))
    if not candidates:
        return None
    highest_step = max(item.state.global_step for item in candidates)
    highest = [item for item in candidates if item.state.global_step == highest_step]
    digests = {item.content_digest for item in highest}
    if len(digests) > 1:
        names = ", ".join(sorted(item.path.name for item in highest))
        raise CheckpointError(
            f"ambiguous checkpoints at step {highest_step}: {names}",
            hint="select the intended directory explicitly with --resume-from",
        )
    final = next((item.path for item in highest if item.path.name == "final"), None)
    return final or sorted((item.path for item in highest), key=lambda path: path.name)[-1]


def _jsonable(value: Any) -> Any:
    """Best-effort conversion of optimizer scalars to JSON."""
    if hasattr(value, "item") and not isinstance(value, (int, float, str, bool)):
        try:
            return value.item()
        except Exception:  # pragma: no cover - defensive
            return str(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
