"""Typed compiler for the pinned verl v0.8 sampled-k1 PG subset."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, ValidationError, model_validator

from miniverl.bridge.contract import VERL_COMMIT, VERL_REPOSITORY, VERL_TAG
from miniverl.bridge.interpolation import reject_interpolation
from miniverl.bridge.opd_v08 import (
    _FIELD_RULES,
    CompatibilityEntry,
    CompiledLocalExecutionPlan,
    FieldClassification,
    VerlOPDDistillationConfig,
    VerlOPDLossConfig,
    VerlOPDV08Profile,
    _canonical_digest,
    _finite,
    _flatten,
    _reject_non_finite_source_numbers,
    _resolve_overrides,
    _Rule,
    _rule,
)
from miniverl.errors import ConfigError

__all__ = [
    "VERL_OPD_PG_K1_V08_PROFILE",
    "compile_verl_pg_k1_v08",
    "load_verl_pg_k1_v08",
    "load_verl_pg_k1_v08_source",
    "pg_compatibility_rule",
    "pg_field_rules_digest",
]

VERL_OPD_PG_K1_V08_PROFILE = "verl-opd-v0.8-single-gpu-pg-k1-v1"


class VerlPGK1LossConfig(VerlOPDLossConfig):
    """PG-only loss fields, isolated from the published direct profile schema."""

    topk: int | None = Field(default=None, ge=1)  # type: ignore[assignment]
    policy_loss_mode: str = "vanilla"
    clip_ratio: float = Field(default=0.2, ge=0, lt=1)
    clip_ratio_low: float | None = Field(default=0.2, ge=0, lt=1)
    clip_ratio_high: float | None = Field(default=0.2, ge=0, lt=1)

    @model_validator(mode="after")
    def _pg_numbers_are_finite(self) -> VerlPGK1LossConfig:
        _finite(self.clip_ratio, "clip_ratio")
        if self.clip_ratio_low is not None:
            _finite(self.clip_ratio_low, "clip_ratio_low")
        if self.clip_ratio_high is not None:
            _finite(self.clip_ratio_high, "clip_ratio_high")
        return self


class VerlPGK1DistillationConfig(VerlOPDDistillationConfig):
    distillation_loss: VerlPGK1LossConfig


class VerlPGK1V08Profile(VerlOPDV08Profile):
    distillation: VerlPGK1DistillationConfig


_PG_FIELD_RULES: dict[str, _Rule] = dict(_FIELD_RULES)
_PG_FIELD_RULES.update(
    {
        "distillation.distillation_loss.loss_mode": _rule(
            "loss.mode", "semantically_conformant", "dedicated pinned sampled k1 estimator"
        ),
        "distillation.distillation_loss.topk": _rule(
            None,
            "informational_only",
            "not applicable because k1 stores only the sampled-token teacher log-probability",
        ),
        "distillation.distillation_loss.use_policy_gradient": _rule(
            "loss.use_policy_gradient", "exact", "must remain enabled for the PG-k1 profile"
        ),
        "distillation.distillation_loss.policy_loss_mode": _rule(
            "loss.policy_loss_mode", "exact", "pinned verl vanilla policy loss"
        ),
        "distillation.distillation_loss.clip_ratio": _rule(
            "loss.clip_ratio", "exact", "pinned vanilla policy-loss clip ratio"
        ),
        "distillation.distillation_loss.clip_ratio_low": _rule(
            "loss.clip_ratio_low", "exact", "pinned vanilla policy-loss lower clip ratio"
        ),
        "distillation.distillation_loss.clip_ratio_high": _rule(
            "loss.clip_ratio_high", "exact", "pinned vanilla policy-loss upper clip ratio"
        ),
        "distillation.distillation_loss.log_prob_min_clamp": _rule(
            None,
            "informational_only",
            "not applied by the sampled k1 estimator; the supported value is null",
        ),
    }
)


def pg_compatibility_rule(field: str) -> dict[str, Any]:
    """Return one immutable PG-profile field rule."""
    try:
        rule = _PG_FIELD_RULES[field]
    except KeyError as exc:
        raise ConfigError(
            f"field {field!r} is not part of profile {VERL_OPD_PG_K1_V08_PROFILE!r}"
        ) from exc
    return {
        "upstream_field": field,
        "local_target": rule.target,
        "classification": rule.classification,
        "reason": rule.reason,
        "semantic_risk": rule.risk,
        "user_confirmation_required": rule.confirmation,
    }


def pg_field_rules_digest() -> str:
    """Bind the PG profile identity to its complete field-rule table."""
    return _canonical_digest([pg_compatibility_rule(field) for field in sorted(_PG_FIELD_RULES)])


def _pg_semantic_blocker(
    path: str, value: Any, *, allow_grouped_samples: bool = False
) -> str | None:
    required = {
        "distillation.enabled": (True, "distillation must be enabled"),
        "distillation.distillation_loss.loss_mode": (
            "k1",
            "only the pinned sampled k1 estimator is supported",
        ),
        "distillation.distillation_loss.use_policy_gradient": (
            True,
            "policy-gradient optimization must be enabled",
        ),
        "distillation.distillation_loss.policy_loss_mode": (
            "vanilla",
            "only the pinned vanilla policy loss is supported",
        ),
        "distillation.distillation_loss.use_task_rewards": (
            False,
            "task-reward mixtures are unsupported",
        ),
        "distillation.distillation_loss.topk": (
            None,
            "top-k targets are not part of sampled k1",
        ),
        "distillation.distillation_loss.log_prob_min_clamp": (
            None,
            "log-probability clamping is not part of the pinned sampled k1 path",
        ),
        "distillation.distillation_loss.clip_ratio": (
            0.2,
            "only the conformance-tested clip_ratio=0.2 is supported",
        ),
        "distillation.distillation_loss.clip_ratio_low": (
            0.2,
            "only the conformance-tested clip_ratio_low=0.2 is supported",
        ),
        "distillation.distillation_loss.clip_ratio_high": (
            0.2,
            "only the conformance-tested clip_ratio_high=0.2 is supported",
        ),
        "actor_rollout_ref.actor.use_kl_loss": (
            False,
            "actor/reference KL loss is outside pure OPD",
        ),
        "algorithm.use_kl_in_reward": (False, "KL reward is outside pure OPD"),
        "actor_rollout_ref.actor.loss_agg_mode": (
            "token-mean",
            "only token-mean aggregation is supported",
        ),
        "actor_rollout_ref.rollout.tensor_model_parallel_size": (
            1,
            "tensor parallelism greater than one is unsupported",
        ),
        "distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size": (
            1,
            "teacher tensor parallelism greater than one is unsupported",
        ),
        "distillation.teacher_models.teacher_model.inference.data_parallel_size": (
            1,
            "teacher data parallelism greater than one is unsupported",
        ),
        "distillation.teacher_models.teacher_model.inference.pipeline_model_parallel_size": (
            1,
            "teacher pipeline parallelism greater than one is unsupported",
        ),
        "trainer.nnodes": (1, "multi-node execution is unsupported"),
    }
    if path == "actor_rollout_ref.rollout.n":
        if allow_grouped_samples and isinstance(value, int) and value >= 1:
            return None
        if value != 1:
            return "one generation per prompt is required by this legacy profile"
    expected = required.get(path)
    if expected is not None and value != expected[0]:
        return expected[1]
    if path == "algorithm.adv_estimator" and value is not None:
        return "external PPO/GRPO advantages are unsupported"
    if path == "trainer.n_gpus_per_node" and value != 1:
        return "more than one local training GPU is unsupported"
    if path == "distillation.nnodes" and value not in {0, 1}:
        return "multi-node teacher execution is unsupported"
    if path == "data.filter_overlong_prompts" and value:
        return "filter_overlong_prompts=true is unsupported; use explicit truncation"
    if (
        path
        in {
            "actor_rollout_ref.actor.fsdp_config.param_offload",
            "actor_rollout_ref.actor.fsdp_config.optimizer_offload",
        }
        and value
    ):
        return "FSDP offload semantics are unsupported by the one-GPU local runtime"
    return None


def compile_verl_pg_k1_v08(
    payload: Mapping[str, Any],
    *,
    override_files: Sequence[Path] = (),
    overrides: Sequence[str] = (),
    trailing_overrides: Sequence[str] = (),
    reinterpretations_accepted: bool = True,
    acceptance_source: str = "library_call",
    require_executable: bool = True,
    allow_grouped_samples: bool = False,
    profile_name: str = VERL_OPD_PG_K1_V08_PROFILE,
) -> CompiledLocalExecutionPlan:
    """Compile the closed sampled-k1 profile into a deterministic local plan."""
    merged = copy.deepcopy(dict(payload))
    reject_interpolation(merged, label="verl PG-k1 OPD config")
    records = _resolve_overrides(
        merged,
        override_files=override_files,
        overrides=overrides,
        trailing_overrides=trailing_overrides,
    )
    reject_interpolation(merged, label="resolved verl PG-k1 OPD config")

    teachers = (
        merged.get("distillation", {}).get("teacher_models", {})
        if isinstance(merged.get("distillation"), Mapping)
        else {}
    )
    if isinstance(teachers, Mapping) and set(teachers) != {"teacher_model"}:
        raise ConfigError(
            "verl PG-k1 OPD config is not executable: multi-teacher routing is unsupported; "
            f"found {sorted(map(str, teachers))}"
        )

    flat = _flatten(merged)
    _reject_non_finite_source_numbers(flat)
    unknown = sorted(path for path in flat if path not in _PG_FIELD_RULES)
    if unknown:
        raise ConfigError(
            f"verl PG-k1 OPD config is not executable: unsupported field {unknown[0]!r}",
            hint="the profile accepts only its documented resolved verl v0.8 subset",
        )
    try:
        source = VerlPGK1V08Profile.model_validate(merged)
    except ValidationError as exc:
        finite = any("finite" in error.get("msg", "").lower() for error in exc.errors())
        message = "verl PG-k1 numeric fields must be finite" if finite else "invalid PG-k1 profile"
        raise ConfigError(message, hint=str(exc)) from exc

    compatibility: list[CompatibilityEntry] = []
    blockers: list[str] = []
    for path, value in sorted(flat.items()):
        rule = _PG_FIELD_RULES[path]
        blocker = _pg_semantic_blocker(
            path,
            value,
            allow_grouped_samples=allow_grouped_samples,
        )
        if blocker:
            blockers.append(path)
            classification: FieldClassification = "unsupported"
            reason = blocker
            risk: Literal["none", "low", "medium", "high"] = "high"
        else:
            classification = rule.classification
            reason = rule.reason
            risk = rule.risk
        if path == "trainer.total_epochs" and source.trainer.total_training_steps is not None:
            classification = "informational_only"
            reason = "recorded but superseded by the explicit total_training_steps cap"
            risk = "none"
        if (
            path
            in {
                "actor_rollout_ref.model.lora_adapter_path",
                "miniverl.teacher_adapter.path",
                "miniverl.teacher_adapter.revision",
            }
            and value is None
        ):
            classification = "informational_only"
            reason = "null records that no existing adapter input is configured"
            risk = "none"
        confirmation_required = (rule.confirmation and classification != "informational_only") or (
            classification == "locally_reinterpreted" and risk == "high"
        )
        compatibility.append(
            CompatibilityEntry(
                upstream_field=path,
                source_value=value,
                local_target=rule.target,
                classification=classification,
                reason=reason,
                semantic_risk=risk,
                user_confirmation_required=confirmation_required,
                executable=blocker is None,
            )
        )

    executable = not blockers
    local_execution = {
        "device_count": 1,
        "distributed_execution": False,
        "compiler_scope": "config_semantics_only",
        "model_weights_loaded": False,
        "roles": ["actor_rollout", "teacher_scoring", "actor_update", "artifact"],
        "runtime_mode": source.miniverl.runtime.mode,
        "student_model": source.actor_rollout_ref.model.path,
        "teacher_model": source.distillation.teacher_models.teacher_model.model_path,
        "loss_mode": "k1",
        "loss_reduction": "token-mean",
        "task_rewards": False,
        "policy_gradient": True,
        "policy_loss_mode": "vanilla",
        "teacher_target": "sampled_token_log_probability",
        "samples_per_prompt": source.actor_rollout_ref.rollout.n,
        "group_semantics": "independent_current_policy_trajectories",
        "trajectory_schema_version": 3 if allow_grouped_samples else 2,
    }
    high_risk = [item.upstream_field for item in compatibility if item.user_confirmation_required]
    common = {
        "profile": profile_name,
        "upstream": {"repository": VERL_REPOSITORY, "tag": VERL_TAG, "commit": VERL_COMMIT},
        "source_digest": _canonical_digest(merged),
        "source_leaf_fields": sorted(flat),
        "source": source,
        "overrides": records,
        "compatibility": compatibility,
        "reinterpretation_acceptance": {
            "required_fields": high_risk,
            "accepted": reinterpretations_accepted or not high_risk,
            "source": acceptance_source if high_risk else "not_required",
        },
        "local_execution": local_execution,
        "executable": executable,
    }
    digest_payload = {
        key: value.model_dump(mode="json") if isinstance(value, BaseModel) else value
        for key, value in common.items()
    }
    digest_payload["overrides"] = [item.model_dump(mode="json") for item in records]
    digest_payload["compatibility"] = [item.model_dump(mode="json") for item in compatibility]
    plan = CompiledLocalExecutionPlan(
        **common,
        compiled_digest=_canonical_digest(digest_payload),
    )
    if require_executable and not executable:
        raise ConfigError(
            "verl PG-k1 OPD config is not executable: unsupported semantics in "
            + ", ".join(blockers)
        )
    return plan


def load_verl_pg_k1_v08(
    path: Path,
    *,
    override_files: Sequence[Path] = (),
    overrides: Sequence[str] = (),
    trailing_overrides: Sequence[str] = (),
    accept_local_reinterpretations: bool = False,
    require_executable: bool = True,
    allow_grouped_samples: bool = False,
    profile_name: str = VERL_OPD_PG_K1_V08_PROFILE,
) -> CompiledLocalExecutionPlan:
    """Load one resolved YAML document without executing interpolation."""
    if path.suffix.lower() in {".sh", ".bash", ".ps1", ".cmd", ".bat"}:
        raise ConfigError("verl PG-k1 input must be resolved YAML, not a shell script")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot read resolved verl PG-k1 YAML {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ConfigError("verl PG-k1 YAML must contain one mapping")
    return compile_verl_pg_k1_v08(
        payload,
        override_files=override_files,
        overrides=overrides,
        trailing_overrides=trailing_overrides,
        reinterpretations_accepted=accept_local_reinterpretations,
        acceptance_source="cli_flag" if accept_local_reinterpretations else "not_accepted",
        require_executable=require_executable,
        allow_grouped_samples=allow_grouped_samples,
        profile_name=profile_name,
    )


def load_verl_pg_k1_v08_source(
    source: str | Path,
    *,
    override_files: Sequence[Path] = (),
    overrides: Sequence[str] = (),
    trailing_overrides: Sequence[str] = (),
    accept_local_reinterpretations: bool = False,
    require_executable: bool = True,
    allow_grouped_samples: bool = False,
    profile_name: str = VERL_OPD_PG_K1_V08_PROFILE,
) -> CompiledLocalExecutionPlan:
    """Load the PG profile from an explicit resolved YAML path."""
    return load_verl_pg_k1_v08(
        Path(source),
        override_files=override_files,
        overrides=overrides,
        trailing_overrides=trailing_overrides,
        accept_local_reinterpretations=accept_local_reinterpretations,
        require_executable=require_executable,
        allow_grouped_samples=allow_grouped_samples,
        profile_name=profile_name,
    )
