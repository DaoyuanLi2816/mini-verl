"""Pickle-free checkpointing and resume.

A checkpoint directory holds three files::

    checkpoints/step-000012/
      adapter.safetensors      # trainable weights only
      optimizer.safetensors    # optimizer moment tensors
      state.json               # everything that is not a tensor

Tensors go through safetensors and structure goes through JSON, so loading a
checkpoint cannot execute code.  ``torch.save`` is never used.

What is restored
----------------
Trainable weights, optimizer moments and hyper-parameter groups, the LR
scheduler, the gradient scaler, the global step, the policy version, the task
sampler position and every RNG state.  ``tests/integration/test_resume.py``
runs the same schedule with and without an interruption and asserts the final
parameters match bit-for-bit.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from miniverl.errors import CheckpointError
from miniverl.utils.runs import write_text
from miniverl.utils.seeding import RngSnapshot

__all__ = ["CheckpointState", "save_checkpoint", "load_checkpoint", "list_checkpoints"]

_ADAPTER = "adapter.safetensors"
_OPTIMIZER = "optimizer.safetensors"
_STATE = "state.json"
CHECKPOINT_SCHEMA_VERSION = 1


@dataclass
class CheckpointState:
    """Non-tensor training state."""

    schema_version: int = CHECKPOINT_SCHEMA_VERSION
    miniverl_version: str = ""
    global_step: int = 0
    policy_version: int = 0
    cycle: int = 0
    task_cursor: int = 0
    scheduler: dict[str, Any] = field(default_factory=dict)
    scaler: dict[str, Any] | None = None
    optimizer_param_groups: list[dict[str, Any]] = field(default_factory=list)
    optimizer_state_keys: list[str] = field(default_factory=list)
    optimizer_scalars: dict[str, Any] = field(default_factory=dict)
    rng: dict[str, Any] = field(default_factory=dict)
    config_digest: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly view."""
        return {
            "schema_version": self.schema_version,
            "miniverl_version": self.miniverl_version,
            "global_step": self.global_step,
            "policy_version": self.policy_version,
            "cycle": self.cycle,
            "task_cursor": self.task_cursor,
            "scheduler": self.scheduler,
            "scaler": self.scaler,
            "optimizer_param_groups": self.optimizer_param_groups,
            "optimizer_state_keys": self.optimizer_state_keys,
            "optimizer_scalars": self.optimizer_scalars,
            "rng": self.rng,
            "config_digest": self.config_digest,
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CheckpointState:
        """Rebuild from :meth:`to_dict` output."""
        version = payload.get("schema_version")
        if version != CHECKPOINT_SCHEMA_VERSION:
            raise CheckpointError(
                f"checkpoint schema_version {version!r} is not readable by this build "
                f"(expected {CHECKPOINT_SCHEMA_VERSION})"
            )
        return cls(**payload)


def _split_optimizer_state(
    optimizer: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], list[str]]:
    """Separate an optimizer state dict into tensors, groups and scalars."""
    import torch

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
    groups = [dict(g) for g in state.get("param_groups", [])]
    return tensors, groups, scalars, keys


def _rebuild_optimizer_state(
    tensors: dict[str, Any],
    scalars: dict[str, Any],
    keys: list[str],
    groups: list[dict[str, Any]],
) -> dict[str, Any]:
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


def save_checkpoint(
    directory: str | Path,
    *,
    trainable_state: dict[str, Any],
    optimizer: Any,
    state: CheckpointState,
    rng: RngSnapshot,
) -> Path:
    """Write a complete, resumable checkpoint."""
    from safetensors.torch import save_file

    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)

    if trainable_state:
        save_file(
            {k: v.contiguous() for k, v in trainable_state.items()},
            str(target / _ADAPTER),
            metadata={"miniverl": "adapter"},
        )
    tensors, groups, scalars, keys = _split_optimizer_state(optimizer)
    if tensors:
        save_file(tensors, str(target / _OPTIMIZER), metadata={"miniverl": "optimizer"})
    state.optimizer_param_groups = groups
    state.optimizer_scalars = {k: _jsonable(v) for k, v in scalars.items()}
    state.optimizer_state_keys = keys
    state.rng = rng.to_dict()
    write_text(
        target / _STATE,
        json.dumps(state.to_dict(), indent=2, sort_keys=True, default=_jsonable) + "\n",
    )
    return target


def load_checkpoint(
    directory: str | Path,
    *,
    backend: Any,
    optimizer: Any,
    device: str = "cpu",
    include_optimizer: bool = True,
    include_rng: bool = True,
) -> CheckpointState:
    """Restore weights, optimizer state and RNG from a checkpoint directory.

    ``include_optimizer=False`` loads *weights only*.  The benchmark harness
    uses that so every arm starts from identical parameters without inheriting
    the cold start's Adam momentum, which would otherwise advantage whichever
    arm most resembles the cold start.
    """
    from safetensors.torch import load_file

    from miniverl.utils.seeding import restore_rng

    target = Path(directory)
    state_path = target / _STATE
    if not state_path.is_file():
        raise CheckpointError(
            f"{target} is not a miniVERL checkpoint (missing {_STATE})",
            hint="pass a directory such as runs/<run-id>/checkpoints/step-000010",
        )
    state = CheckpointState.from_dict(json.loads(state_path.read_text(encoding="utf-8")))

    adapter_path = target / _ADAPTER
    if adapter_path.is_file():
        backend.load_trainable_state_dict(load_file(str(adapter_path), device=device))

    if include_optimizer:
        optimizer_path = target / _OPTIMIZER
        tensors = load_file(str(optimizer_path), device=device) if optimizer_path.is_file() else {}
        if state.optimizer_state_keys:
            optimizer.load_state_dict(
                _rebuild_optimizer_state(
                    tensors,
                    state.optimizer_scalars,
                    state.optimizer_state_keys,
                    state.optimizer_param_groups,
                )
            )
    if include_rng and state.rng:
        restore_rng(RngSnapshot.from_dict(state.rng))
    return state


def list_checkpoints(directory: str | Path) -> list[Path]:
    """Sorted checkpoint directories under ``directory``."""
    root = Path(directory)
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir() and (p / _STATE).is_file())


def latest_checkpoint(directory: str | Path) -> Path | None:
    """Most recent checkpoint, or ``None``."""
    entries = list_checkpoints(directory)
    return entries[-1] if entries else None


def _jsonable(value: Any) -> Any:
    """Best-effort conversion of optimizer scalars to JSON."""
    if hasattr(value, "item") and not isinstance(value, (int, float, str, bool)):
        try:
            return value.item()
        except Exception:  # pragma: no cover - defensive
            return str(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value
