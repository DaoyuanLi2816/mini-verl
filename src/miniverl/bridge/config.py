"""Fail-closed import for the pinned, documented verl profile subset."""

from __future__ import annotations

import hashlib
import math
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

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

FieldClass = Literal[
    "exact",
    "derived",
    "informational_only",
    "requires_user_confirmation",
    "unsupported",
]

_FIELD_RULES: dict[str, tuple[str | None, FieldClass, str]] = {
    "data.train_files": (
        None,
        "informational_only",
        "miniVERL trains against an explicitly selected local ToolEnvironment, not this Parquet path",
    ),
    "data.val_files": (
        None,
        "informational_only",
        "miniVERL evaluates an explicitly selected local ToolEnvironment, not this Parquet path",
    ),
    "data.prompt_key": (
        None,
        "informational_only",
        "the selected miniVERL environment owns prompt construction",
    ),
    "data.max_prompt_length": (
        "rollout.max_total_tokens",
        "derived",
        "combined with max_response_length; miniVERL has a total trajectory-token bound",
    ),
    "data.max_response_length": (
        "rollout.max_new_tokens_per_turn",
        "exact",
        "copied as the per-turn generation bound",
    ),
    "data.seed": ("run.seed", "exact", "copied without a unit change"),
    "actor_rollout_ref.model.path": (
        "models.student.model_id",
        "exact",
        "copied as the student model identity",
    ),
    "actor_rollout_ref.model.enable_gradient_checkpointing": (
        "models.student.gradient_checkpointing",
        "exact",
        "copied as a model construction option",
    ),
    "actor_rollout_ref.actor.optim.lr": (
        "train.learning_rate",
        "exact",
        "copied as the optimizer learning rate",
    ),
    "trainer.save_freq": (
        "train.save_every_cycles",
        "requires_user_confirmation",
        "verl frequency units are not proven equivalent to miniVERL cycles",
    ),
    "trainer.test_freq": (
        "train.eval_every_cycles",
        "requires_user_confirmation",
        "verl frequency units are not proven equivalent to miniVERL cycles",
    ),
    "trainer.project_name": (
        "run.name",
        "derived",
        "combined with trainer.experiment_name",
    ),
    "trainer.experiment_name": (
        "run.name",
        "derived",
        "combined with trainer.project_name",
    ),
    "trainer.total_epochs": (
        "train.cycles",
        "requires_user_confirmation",
        "epochs and miniVERL continuation cycles are not proven equivalent",
    ),
    "trainer.logger": (None, "informational_only", "not used by the local runtime"),
    "trainer.resume_mode": (None, "informational_only", "not used by the import"),
    "trainer.default_local_dir": (
        None,
        "informational_only",
        "not copied because output provenance is rooted at the requested destination",
    ),
}

_LOSS_PROFILES: dict[str, dict[str, str]] = {
    "topk-tail-reverse-kl": {
        "mode": "bucketed_topk_tail",
        "divergence": "reverse_kl",
    },
    "topk-tail-forward-kl": {
        "mode": "bucketed_topk_tail",
        "divergence": "forward_kl",
    },
    "exact-reverse-kl": {"mode": "exact_full_vocab", "divergence": "reverse_kl"},
    "exact-forward-kl": {"mode": "exact_full_vocab", "divergence": "forward_kl"},
}
_SCHEDULE_MAPPING = "epochs-as-cycles"


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
    if isinstance(value, str):
        stripped = value.strip()
        if "${" in stripped:
            raise ConfigError(
                f"verl field {field} contains an unresolved interpolation",
                hint="pass a fully resolved, documented profile before importing",
            )
        try:
            parsed = float(stripped)
        except ValueError as exc:
            raise ConfigError(
                f"verl field {field} must be a finite positive number; got {value!r}"
            ) from exc
    elif isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"verl field {field} must be a finite positive number")
    else:
        parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise ConfigError(f"verl field {field} must be a finite positive number")
    return parsed


def _atomic_yaml(path: Path, payload: dict[str, Any]) -> bytes:
    rendered = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=100).encode(
        "utf-8"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_bytes(rendered)
    temporary.replace(path)
    return rendered


def _source_identity() -> dict[str, str]:
    return {"repository": VERL_REPOSITORY, "tag": VERL_TAG, "commit": VERL_COMMIT}


def _classify(flat: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    classified: dict[str, dict[str, Any]] = {}
    for path, value in sorted(flat.items()):
        target, classification, reason = _FIELD_RULES.get(
            path,
            (None, "unsupported", "outside the documented resolved-profile subset"),
        )
        classified[path] = {
            "target": target,
            "classification": classification,
            "value": portable_payload(value),
            "reason": reason,
        }
    return classified


def _required_inputs(
    *,
    student_model: str,
    environment: str | None,
    teacher_model: str | None,
    teacher_adapter: str | None,
    loss_profile: str | None,
    schedule_mapping: str | None,
) -> list[dict[str, str]]:
    required: list[dict[str, str]] = []
    if not environment:
        required.append(
            {
                "field": "environment",
                "reason": "Parquet file names do not identify a miniVERL ToolEnvironment",
                "supply": "--environment <registered-environment>",
            }
        )
    teacher_is_unqualified_same_base = teacher_model == student_model and not teacher_adapter
    if (not teacher_model and not teacher_adapter) or teacher_is_unqualified_same_base:
        required.append(
            {
                "field": "teacher_identity",
                "reason": (
                    "the source does not establish a distinct teacher or a same-base teacher adapter"
                ),
                "supply": "--teacher-model <model> and/or --teacher-adapter <path>",
            }
        )
    if not loss_profile:
        required.append(
            {
                "field": "loss_profile",
                "reason": "the source profile does not determine a miniVERL distillation objective",
                "supply": f"--loss-profile <{'|'.join(sorted(_LOSS_PROFILES))}>",
            }
        )
    if not schedule_mapping:
        required.append(
            {
                "field": "schedule_mapping",
                "reason": "verl epochs/frequencies are not proven equivalent to miniVERL cycles",
                "supply": f"--schedule-mapping {_SCHEDULE_MAPPING}",
            }
        )
    return required


def _template(source: Mapping[str, Any], required: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "needs_user_input",
        "note": "This is a non-executable import template, not a miniVERL RunConfig.",
        "source_profile": BRIDGE_PROFILE,
        "source_values": portable_payload(dict(source)),
        "required_user_input": required,
    }


def _generated_recipe(
    source: Mapping[str, Any],
    *,
    environment: str,
    teacher_model: str | None,
    teacher_adapter: str | None,
    loss_profile: str,
    schedule_mapping: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model_id = _get(source, "actor_rollout_ref.model.path")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ConfigError("verl field actor_rollout_ref.model.path is required")
    if loss_profile not in _LOSS_PROFILES:
        raise ConfigError(
            f"unsupported loss profile {loss_profile!r}",
            hint=f"choose one of {', '.join(sorted(_LOSS_PROFILES))}",
        )
    if schedule_mapping != _SCHEDULE_MAPPING:
        raise ConfigError(
            f"unsupported schedule mapping {schedule_mapping!r}",
            hint=f"use --schedule-mapping {_SCHEDULE_MAPPING} to explicitly accept the unit change",
        )
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

    resolved_teacher = teacher_model or model_id
    teacher: dict[str, Any] = {
        "model_id": resolved_teacher,
        "dtype": "auto",
        "quantization": "none",
        "mode": "standard",
    }
    if teacher_adapter:
        teacher["adapter"] = {"path": teacher_adapter, "source": "local"}

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
            "teacher": teacher,
        },
        "environment": {
            "name": environment,
            "params": {},
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
        "loss": {**_LOSS_PROFILES[loss_profile], "top_k": 64},
        "train": {
            "cycles": cycles,
            "learning_rate": learning_rate,
            "save_every_cycles": save_freq,
            "eval_every_cycles": test_freq,
            "opd_freshness": "strict",
        },
        "cache": {"strict_policy_version": True, "reuse_across_policy_versions": False},
    }
    defaults = [
        {
            "field": "environment split sizes",
            "value": {"train": 64, "eval": 32, "test": 32},
            "reason": "the verl subset has file paths but no miniVERL task-pool sizes",
            "source_run_intent": False,
        },
        {
            "field": "selection.selector",
            "value": "all_model_tokens",
            "reason": "the verl subset does not encode miniVERL token provenance selection",
            "source_run_intent": False,
        },
        {
            "field": "loss.top_k",
            "value": 64,
            "reason": "profile constant used only for a top-k + tail loss profile",
            "source_run_intent": False,
        },
    ]
    return recipe, defaults


def import_verl_config(
    source: str | Path,
    *,
    profile: str,
    target_verl: str,
    out: str | Path,
    environment: str | None = None,
    teacher_model: str | None = None,
    teacher_adapter: str | None = None,
    loss_profile: str | None = None,
    schedule_mapping: str | None = None,
) -> dict[str, Any]:
    """Import the resolved profile subset or emit a non-executable template."""
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
    classification = _classify(flat)
    unsupported = sorted(
        path
        for path, decision in classification.items()
        if decision["classification"] == "unsupported"
    )
    report_path = Path(out).parent / "import-report.json"
    common: dict[str, Any] = {
        "schema_version": 2,
        "source_verl": _source_identity(),
        "profile": BRIDGE_PROFILE,
        "input_contract": "resolved documented profile subset; not arbitrary verl YAML",
        "source_config_sha256": _digest_bytes(source_bytes),
        "field_classification": classification,
        "unsupported_fields": unsupported,
        "semantic_conflicts": [],
    }
    if unsupported:
        rejection_report = {
            **common,
            "inserted_defaults": [],
            "required_user_input": [],
            "generated_miniverl_sha256": None,
            "generated_recipe_validated": False,
            "generated_path": None,
            "status": "rejected",
        }
        write_json_atomic(report_path, rejection_report)
        raise ConfigError(
            f"unsupported verl field {unsupported[0]!r} for profile {BRIDGE_PROFILE}",
            hint=(
                "remove algorithm, distributed, rollout-runtime or unknown fields; "
                "inspect the documented resolved-profile whitelist"
            ),
        )

    student_model = _get(payload, "actor_rollout_ref.model.path")
    if not isinstance(student_model, str) or not student_model.strip():
        raise ConfigError("verl field actor_rollout_ref.model.path is required")
    required = _required_inputs(
        student_model=student_model,
        environment=environment,
        teacher_model=teacher_model,
        teacher_adapter=teacher_adapter,
        loss_profile=loss_profile,
        schedule_mapping=schedule_mapping,
    )
    destination = Path(out)
    if required:
        template_path = destination.parent / "imported.template.yaml"
        template = _template(payload, required)
        rendered = yaml.safe_dump(template, sort_keys=False, allow_unicode=True, width=100).encode(
            "utf-8"
        )
        report = {
            **common,
            "inserted_defaults": [],
            "required_user_input": required,
            "user_confirmations": {},
            "generated_miniverl_sha256": _digest_bytes(rendered),
            "generated_recipe_validated": False,
            "generated_path": template_path.name,
            "status": "needs_user_input",
            "claim": "No runnable miniVERL recipe was generated.",
        }
        _atomic_yaml(template_path, template)
        try:
            write_json_atomic(report_path, report)
        except BaseException:
            template_path.unlink(missing_ok=True)
            raise
        return report

    assert environment is not None
    assert loss_profile is not None
    assert schedule_mapping is not None
    recipe, inserted_defaults = _generated_recipe(
        payload,
        environment=environment,
        teacher_model=teacher_model,
        teacher_adapter=teacher_adapter,
        loss_profile=loss_profile,
        schedule_mapping=schedule_mapping,
    )
    try:
        from miniverl.config import RunConfig

        RunConfig.from_mapping(recipe)
    except Exception as exc:
        raise ConfigError(
            f"generated miniVERL recipe failed RunConfig validation: {exc}",
            hint="check the explicit environment, teacher and loss-profile arguments",
        ) from exc
    rendered = yaml.safe_dump(recipe, sort_keys=False, allow_unicode=True, width=100).encode(
        "utf-8"
    )
    report = {
        **common,
        "inserted_defaults": inserted_defaults,
        "required_user_input": [],
        "user_confirmations": {
            "environment": environment,
            "teacher_model": teacher_model,
            "teacher_adapter": teacher_adapter,
            "loss_profile": loss_profile,
            "schedule_mapping": schedule_mapping,
        },
        "generated_miniverl_sha256": _digest_bytes(rendered),
        "generated_recipe_validated": True,
        "generated_path": destination.name,
        "status": "accepted",
        "claim": "Imports only the resolved documented profile subset for pinned verl v0.8.0.",
    }
    _atomic_yaml(destination, recipe)
    try:
        write_json_atomic(report_path, report)
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return report
