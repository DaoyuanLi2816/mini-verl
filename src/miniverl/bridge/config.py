"""Fail-closed import for the pinned verl single-GPU distillation profile."""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from miniverl.bridge.contract import (
    BRIDGE_PROFILE,
    VERL_COMMIT,
    VERL_REPOSITORY,
    VERL_TAG,
    validate_target_verl,
)
from miniverl.errors import ConfigError
from miniverl.utils.privacy import portable_payload
from miniverl.utils.runs import write_json_atomic

__all__ = ["import_verl_config"]

_MAPPED: dict[str, tuple[str | None, str]] = {
    "data.train_files": (None, "bridge_metadata"),
    "data.val_files": (None, "bridge_metadata"),
    "data.prompt_key": (None, "bridge_metadata"),
    "data.max_prompt_length": ("rollout.max_total_tokens", "mapped"),
    "data.max_response_length": ("rollout.max_new_tokens_per_turn", "mapped"),
    "data.seed": ("run.seed", "mapped"),
    "actor_rollout_ref.model.path": ("models.student.model_id", "mapped"),
    "actor_rollout_ref.model.enable_gradient_checkpointing": (
        "models.student.gradient_checkpointing",
        "mapped",
    ),
    "actor_rollout_ref.actor.optim.lr": ("train.learning_rate", "mapped"),
    "trainer.save_freq": ("train.save_every_cycles", "mapped"),
    "trainer.test_freq": ("train.eval_every_cycles", "mapped"),
    "trainer.project_name": ("run.name", "mapped"),
    "trainer.experiment_name": ("run.name", "mapped"),
    "trainer.total_epochs": ("train.cycles", "mapped"),
}

_IGNORED_INFORMATIONAL = {
    "trainer.logger",
    "trainer.resume_mode",
    "trainer.default_local_dir",
}


def _digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _flatten(value: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, Mapping):
            flattened.update(_flatten(item, path))
        else:
            flattened[path] = item
    return flattened


def _get(payload: Mapping[str, Any], path: str, default: Any = None) -> Any:
    current: Any = payload
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def _integer(value: Any, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigError(f"verl field {field} must be an integer >= {minimum}")
    return value


def _positive_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) <= 0:
        raise ConfigError(f"verl field {field} must be a positive number")
    return float(value)


def _atomic_yaml(path: Path, payload: dict[str, Any]) -> bytes:
    rendered = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=100).encode(
        "utf-8"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_bytes(rendered)
    temporary.replace(path)
    return rendered


def _generated_recipe(source: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model_id = _get(source, "actor_rollout_ref.model.path")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ConfigError("verl field actor_rollout_ref.model.path is required")
    project = _get(source, "trainer.project_name", "verl-import")
    experiment = _get(source, "trainer.experiment_name", "profile")
    if not isinstance(project, str) or not isinstance(experiment, str):
        raise ConfigError("verl trainer.project_name and experiment_name must be strings")
    prompt_length = _integer(
        _get(source, "data.max_prompt_length", 512), "data.max_prompt_length", minimum=1
    )
    response_length = _integer(
        _get(source, "data.max_response_length", 128), "data.max_response_length", minimum=1
    )
    seed = _integer(_get(source, "data.seed", 1234), "data.seed")
    cycles = _integer(_get(source, "trainer.total_epochs", 1), "trainer.total_epochs")
    save_freq = _integer(_get(source, "trainer.save_freq", 0), "trainer.save_freq")
    test_freq = _integer(_get(source, "trainer.test_freq", 0), "trainer.test_freq")
    learning_rate = _positive_number(
        _get(source, "actor_rollout_ref.actor.optim.lr", 1e-4),
        "actor_rollout_ref.actor.optim.lr",
    )
    gradient_checkpointing = _get(
        source, "actor_rollout_ref.model.enable_gradient_checkpointing", False
    )
    if not isinstance(gradient_checkpointing, bool):
        raise ConfigError(
            "verl field actor_rollout_ref.model.enable_gradient_checkpointing must be boolean"
        )

    recipe: dict[str, Any] = {
        "schema_version": 1,
        "run": {
            "name": f"{project}-{experiment}",
            "mode": "opd",
            "seed": seed,
            "output_dir": "runs",
            "deterministic": True,
            "tags": ["verl-bridge", BRIDGE_PROFILE],
        },
        "models": {
            "backend": "hf",
            "runtime": "dual_model",
            "device": "auto",
            "student": {
                "model_id": model_id,
                "dtype": "auto",
                "quantization": "none",
                "gradient_checkpointing": gradient_checkpointing,
                "lora": {"enabled": True},
            },
            "teacher": {
                "model_id": model_id,
                "dtype": "auto",
                "quantization": "none",
                "mode": "standard",
            },
        },
        "environment": {
            "name": "calculator",
            "params": {"protocol_version": "v2", "prompt_style": "compact"},
            "train_tasks": 64,
            "eval_tasks": 32,
            "test_tasks": 32,
            "split_seed": seed,
        },
        "rollout": {
            "max_new_tokens_per_turn": response_length,
            "max_total_tokens": prompt_length + response_length,
        },
        "selection": {"selector": "all_model_tokens"},
        "loss": {
            "mode": "bucketed_topk_tail",
            "divergence": "reverse_kl",
            "top_k": 64,
        },
        "train": {
            "cycles": cycles,
            "learning_rate": learning_rate,
            "save_every_cycles": save_freq,
            "eval_every_cycles": test_freq,
            "opd_freshness": "strict",
        },
        "cache": {
            "strict_policy_version": True,
            "reuse_across_policy_versions": False,
        },
    }
    defaults = [
        {
            "field": "models.teacher",
            "value": "policy-conditioned same-base teacher",
            "reason": "the profile imports no separate teacher identity",
        },
        {
            "field": "environment",
            "value": "calculator protocol-v2 scaffold",
            "reason": "verl prompt data does not identify a miniVERL tool environment",
        },
        {
            "field": "loss",
            "value": "reverse_kl top-k-plus-tail",
            "reason": "PPO/GRPO semantics are intentionally outside this profile",
        },
    ]
    return recipe, defaults


def import_verl_config(
    source: str | Path,
    *,
    profile: str,
    target_verl: str,
    out: str | Path,
) -> dict[str, Any]:
    """Import only the documented profile and emit a complete decision report."""
    if profile != BRIDGE_PROFILE:
        raise ConfigError(
            f"unsupported verl bridge profile {profile!r}", hint=f"use --profile {BRIDGE_PROFILE}"
        )
    validate_target_verl(target_verl)
    source_path = Path(source)
    try:
        source_bytes = source_path.read_bytes()
        payload = yaml.safe_load(source_bytes)
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot read verl config {source_path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ConfigError("verl config must contain one YAML mapping")

    flat = _flatten(payload)
    unsupported = sorted(
        path for path in flat if path not in _MAPPED and path not in _IGNORED_INFORMATIONAL
    )
    if unsupported:
        rejection_report = {
            "schema_version": 1,
            "source_verl": {
                "repository": VERL_REPOSITORY,
                "tag": VERL_TAG,
                "commit": VERL_COMMIT,
            },
            "profile": BRIDGE_PROFILE,
            "source_config_sha256": _digest_bytes(source_bytes),
            "mapped_fields": {},
            "ignored_informational_fields": [],
            "unsupported_fields": unsupported,
            "semantic_conflicts": [],
            "inserted_defaults": [],
            "generated_miniverl_sha256": None,
            "status": "rejected",
        }
        write_json_atomic(Path(out).parent / "import-report.json", rejection_report)
        raise ConfigError(
            f"unsupported verl field {unsupported[0]!r} for profile {BRIDGE_PROFILE}",
            hint="remove algorithm, distributed, rollout-runtime or unknown fields; inspect the documented whitelist",
        )

    recipe, inserted_defaults = _generated_recipe(payload)
    destination = Path(out)
    rendered = yaml.safe_dump(recipe, sort_keys=False, allow_unicode=True, width=100).encode(
        "utf-8"
    )
    mapped_fields = {
        path: {
            "target": target,
            "disposition": disposition,
            "value": portable_payload(flat[path]),
        }
        for path, (target, disposition) in _MAPPED.items()
        if path in flat
    }
    ignored = [
        {"field": path, "value": portable_payload(flat[path])}
        for path in sorted(_IGNORED_INFORMATIONAL.intersection(flat))
    ]
    report: dict[str, Any] = {
        "schema_version": 1,
        "source_verl": {
            "repository": VERL_REPOSITORY,
            "tag": VERL_TAG,
            "commit": VERL_COMMIT,
        },
        "profile": BRIDGE_PROFILE,
        "source_config_sha256": _digest_bytes(source_bytes),
        "mapped_fields": mapped_fields,
        "ignored_informational_fields": ignored,
        "unsupported_fields": [],
        "semantic_conflicts": [],
        "inserted_defaults": inserted_defaults,
        "generated_miniverl_sha256": _digest_bytes(rendered),
        "status": "accepted",
        "claim": (
            "Imports the documented single-gpu-online-distillation-v1 subset of pinned verl v0.8.0."
        ),
    }

    _atomic_yaml(destination, recipe)
    try:
        write_json_atomic(destination.parent / "import-report.json", report)
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return report
