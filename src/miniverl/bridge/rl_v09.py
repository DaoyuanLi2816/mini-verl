"""Versioned compiler from a resolved verl v0.9 RL subset to one-GPU miniVERL.

The compiler preserves experiment semantics that miniVERL implements and
classifies physical distributed settings separately.  It never imports Python
objects named by an input config and never guesses a reward implementation.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final, Literal

import yaml

from miniverl.algorithms.contract import (
    ADVANTAGE_IMPLEMENTATION_VERSION,
    UPSTREAM_VERL_COMMIT,
    UPSTREAM_VERL_TAG,
)
from miniverl.bridge.interpolation import audit_interpolation
from miniverl.bridge.publish import (
    DEFAULT_LOCK_TIMEOUT,
    OutputTransaction,
    import_output_targets,
    reject_source_output_alias,
)
from miniverl.errors import ConfigError
from miniverl.utils.privacy import portable_payload

__all__ = [
    "VERL_RL_V09_PROFILE",
    "VERL_RL_V09_PPO_PROFILE",
    "publish_imported_verl_rl_v09",
    "rl_v09_field_rules_digest",
]

VERL_RL_V09_PROFILE: Final = "verl-rl-v0.9-single-gpu-v1"
VERL_RL_V09_PPO_PROFILE: Final = "verl-rl-v0.9-single-gpu-v2"
VERL_REPOSITORY: Final = "https://github.com/verl-project/verl"

FieldClassification = Literal[
    "exact",
    "semantically_conformant",
    "locally_lowered",
    "informational_only",
    "distributed_only",
    "not_implemented",
    "unsupported",
]


def _rule(
    target: str | None,
    classification: FieldClassification,
    reason: str,
) -> tuple[str | None, FieldClassification, str]:
    return target, classification, reason


_RULES_V1: dict[str, tuple[str | None, FieldClassification, str]] = {
    "data.train_files": _rule("source.train_files", "exact", "same Parquet inputs"),
    "data.val_files": _rule("source.val_files", "exact", "same Parquet inputs"),
    "data.prompt_key": _rule("source.prompt_key", "exact", "same Parquet column"),
    "data.train_batch_size": _rule(
        "train.rollouts_per_cycle", "semantically_conformant", "same logical prompts per rollout"
    ),
    "data.max_prompt_length": _rule("source.max_prompt_length", "exact", "same token bound"),
    "data.max_response_length": _rule("source.max_response_length", "exact", "same token bound"),
    "data.truncation": _rule("source.truncation", "semantically_conformant", "same policy"),
    "data.shuffle": _rule("source.shuffle", "semantically_conformant", "same data policy"),
    "data.seed": _rule("source.seed", "exact", "same seed"),
    "actor_rollout_ref.model.path": _rule(
        "models.student.model_id", "exact", "same actor checkpoint identity"
    ),
    "actor_rollout_ref.model.enable_gradient_checkpointing": _rule(
        "models.student.gradient_checkpointing",
        "semantically_conformant",
        "same activation-memory technique",
    ),
    "actor_rollout_ref.model.lora_rank": _rule(
        "models.student.lora.r", "semantically_conformant", "same PEFT rank when enabled"
    ),
    "actor_rollout_ref.model.lora_alpha": _rule(
        "models.student.lora.alpha", "semantically_conformant", "same PEFT scale when enabled"
    ),
    "actor_rollout_ref.model.target_modules": _rule(
        "models.student.lora.target_modules",
        "semantically_conformant",
        "same PEFT target names when enabled",
    ),
    "actor_rollout_ref.actor.optim.lr": _rule(
        "train.learning_rate", "exact", "same optimizer learning rate"
    ),
    "actor_rollout_ref.actor.optim.weight_decay": _rule(
        "train.weight_decay", "exact", "same optimizer coefficient"
    ),
    "actor_rollout_ref.actor.optim.lr_warmup_steps": _rule(
        "train.warmup_steps", "exact", "same optimizer-step count"
    ),
    "actor_rollout_ref.actor.ppo_mini_batch_size": _rule(
        "train.gradient_accumulation_steps",
        "semantically_conformant",
        "one logical upstream mini-batch is physically microbatched on one GPU",
    ),
    "actor_rollout_ref.actor.ppo_epochs": _rule(
        "train.opd_freshness", "exact", "one fresh-policy update pass is supported"
    ),
    "actor_rollout_ref.actor.loss_agg_mode": _rule(
        "loss.aggregation", "exact", "token-mean aggregation"
    ),
    "actor_rollout_ref.actor.clip_ratio": _rule(
        "loss.clip_ratio", "exact", "same vanilla policy clip"
    ),
    "actor_rollout_ref.actor.clip_ratio_low": _rule(
        "loss.clip_ratio_low", "exact", "same lower clip"
    ),
    "actor_rollout_ref.actor.clip_ratio_high": _rule(
        "loss.clip_ratio_high", "exact", "same upper clip"
    ),
    "actor_rollout_ref.actor.clip_ratio_c": _rule(
        "loss.clip_ratio_c", "exact", "same dual-clip coefficient"
    ),
    "actor_rollout_ref.actor.use_kl_loss": _rule(
        None, "not_implemented", "RL actor/reference KL is not connected"
    ),
    "actor_rollout_ref.actor.entropy_coeff": _rule(
        None, "not_implemented", "entropy regularization is not connected to the local objective"
    ),
    "actor_rollout_ref.rollout.name": _rule(
        "rollout.backend",
        "locally_lowered",
        "distributed engine selection becomes a validated local HF backend",
    ),
    "actor_rollout_ref.rollout.n": _rule(
        "rollout.samples_per_prompt", "exact", "same grouped sample count"
    ),
    "actor_rollout_ref.rollout.temperature": _rule(
        "rollout.temperature", "exact", "same behavior-policy temperature"
    ),
    "actor_rollout_ref.rollout.top_p": _rule("rollout.top_p", "exact", "same nucleus threshold"),
    "actor_rollout_ref.rollout.top_k": _rule("rollout.top_k", "exact", "same top-k bound"),
    "actor_rollout_ref.rollout.tensor_model_parallel_size": _rule(
        None, "distributed_only", "tensor parallel placement is not part of one-GPU semantics"
    ),
    "actor_rollout_ref.rollout.gpu_memory_utilization": _rule(
        "memory planner",
        "locally_lowered",
        "recorded as an upstream engine hint; local placement is planned separately",
    ),
    "algorithm.adv_estimator": _rule(
        "algorithm.name", "semantically_conformant", "pinned estimator implementation"
    ),
    "algorithm.norm_adv_by_std_in_grpo": _rule(
        "algorithm.name", "semantically_conformant", "false selects Dr.GRPO centering"
    ),
    "algorithm.gamma": _rule("algorithm.gamma", "exact", "same discount"),
    "algorithm.lam": _rule("algorithm.lam", "exact", "same GAE coefficient where applicable"),
    "algorithm.use_kl_in_reward": _rule(
        "algorithm.kl_coef", "semantically_conformant", "same sampled-token KL reward penalty"
    ),
    "algorithm.kl_penalty": _rule(
        "algorithm.kl_penalty", "semantically_conformant", "same pinned estimator"
    ),
    "algorithm.kl_ctrl.type": _rule(
        "algorithm.kl_coef", "semantically_conformant", "fixed coefficient is supported"
    ),
    "algorithm.kl_ctrl.kl_coef": _rule(
        "algorithm.kl_coef", "exact", "same fixed reference-KL coefficient"
    ),
    "trainer.total_training_steps": _rule(
        "train.cycles", "exact", "one local cycle is one upstream rollout iteration"
    ),
    "trainer.total_epochs": _rule(
        None,
        "informational_only",
        "recorded but not used when total_training_steps supplies the exact optimizer-step cap",
    ),
    "trainer.project_name": _rule("run.tags", "semantically_conformant", "same provenance label"),
    "trainer.experiment_name": _rule("run.name", "exact", "same run label"),
    "trainer.save_freq": _rule(
        "train.save_every_cycles", "exact", "same rollout-iteration interval"
    ),
    "trainer.test_freq": _rule(
        "train.eval_every_cycles", "exact", "same rollout-iteration interval"
    ),
    "trainer.logger": _rule(None, "informational_only", "external logging is not activated"),
    "trainer.n_gpus_per_node": _rule(
        None, "distributed_only", "resource placement lowers to one local CUDA device"
    ),
    "trainer.nnodes": _rule(
        None, "distributed_only", "resource placement lowers to one local process"
    ),
    "miniverl.reward.provider": _rule(
        "reward.provider", "exact", "explicit trusted local reward implementation"
    ),
    "miniverl.actor.lora_enabled": _rule(
        "models.student.lora.enabled", "exact", "explicit local parameterization choice"
    ),
    "miniverl.actor.revision": _rule(
        "models.student.revision", "exact", "immutable actor snapshot revision"
    ),
    "miniverl.actor.tokenizer_revision": _rule(
        "models.student.tokenizer_revision", "exact", "immutable tokenizer snapshot revision"
    ),
    "miniverl.actor.dtype": _rule("models.student.dtype", "exact", "local runtime choice"),
    "miniverl.actor.quantization": _rule(
        "models.student.quantization", "exact", "local runtime choice"
    ),
    "miniverl.rollout.backend": _rule("rollout.backend", "exact", "local runtime choice"),
    "miniverl.memory.strategy": _rule("memory.strategy", "exact", "local placement choice"),
    "miniverl.reference.adapter_path": _rule(
        "models.reference.adapter.path", "exact", "explicit frozen reference adapter"
    ),
    "miniverl.reference.adapter_source": _rule(
        "models.reference.adapter.source", "exact", "local or Hub artifact source"
    ),
    "miniverl.reference.adapter_revision": _rule(
        "models.reference.adapter.revision", "exact", "immutable Hub artifact revision"
    ),
    "miniverl.update.trajectory_batch_size": _rule(
        "train.trajectory_batch_size", "exact", "physical-only one-GPU batch size"
    ),
}

_RULES_V2 = dict(_RULES_V1)
_RULES_V2.update(
    {
        "actor_rollout_ref.actor.ppo_epochs": _rule(
            "algorithm.actor_ppo_epochs", "exact", "same actor passes over each rollout batch"
        ),
        "actor_rollout_ref.actor.use_kl_loss": _rule(
            "algorithm.actor_kl_coef", "semantically_conformant", "same sampled-token actor KL"
        ),
        "actor_rollout_ref.actor.kl_loss_coef": _rule(
            "algorithm.actor_kl_coef", "exact", "same actor-loss KL coefficient"
        ),
        "actor_rollout_ref.actor.kl_loss_type": _rule(
            "algorithm.actor_kl_penalty", "semantically_conformant", "same pinned KL estimator"
        ),
        "actor_rollout_ref.actor.entropy_coeff": _rule(
            "algorithm.entropy_coeff", "exact", "same token-mean entropy regularization"
        ),
        "critic.enable": _rule("critic.enabled", "exact", "explicit trainable value role"),
        "critic.model.path": _rule(
            "critic independent backbone", "semantically_conformant", "same pretrained identity"
        ),
        "critic.optim.lr": _rule("critic.learning_rate", "exact", "same optimizer learning rate"),
        "critic.optim.weight_decay": _rule(
            "critic.weight_decay", "exact", "same optimizer coefficient"
        ),
        "critic.optim.lr_warmup_steps": _rule(
            "critic.warmup_steps", "exact", "same critic optimizer-step count"
        ),
        "critic.ppo_mini_batch_size": _rule(
            "train.gradient_accumulation_steps",
            "semantically_conformant",
            "same logical minibatch; physical microbatching remains independent",
        ),
        "critic.ppo_epochs": _rule(
            "critic.ppo_epochs", "exact", "same critic passes over each rollout batch"
        ),
        "critic.cliprange_value": _rule(
            "algorithm.cliprange_value", "exact", "same old-value clipping interval"
        ),
        "miniverl.critic.lora_enabled": _rule(
            "critic independent backbone",
            "semantically_conformant",
            "same explicit PEFT parameterization as the actor",
        ),
        "reward_model.enable": _rule(
            "reward.provider", "semantically_conformant", "enables the trained local reward role"
        ),
        "reward_model.model.path": _rule(
            "reward.model.model_id", "exact", "same trained reward-model identity"
        ),
        "reward_model.model.trust_remote_code": _rule(
            "reward.model.trust_remote_code", "exact", "same explicit code-trust choice"
        ),
        "miniverl.reward.revision": _rule(
            "reward.model.revision", "exact", "immutable reward-model snapshot"
        ),
        "miniverl.reward.tokenizer_revision": _rule(
            "reward.model.tokenizer_revision", "exact", "immutable reward tokenizer snapshot"
        ),
        "miniverl.reward.positive_class_index": _rule(
            "reward.model.positive_class_index", "exact", "same selected reward label"
        ),
        "miniverl.reward.max_length": _rule(
            "reward.model.max_length", "locally_lowered", "local deterministic token bound"
        ),
        "miniverl.reward.batch_size": _rule(
            "reward.model.batch_size", "locally_lowered", "physical one-GPU inference batch"
        ),
        "miniverl.reward.timeout_seconds": _rule(
            "reward.model.timeout_seconds", "locally_lowered", "local fail-closed timeout"
        ),
        "miniverl.reward.offload_between_phases": _rule(
            "reward.model.offload_between_phases",
            "locally_lowered",
            "single-GPU temporal reward-role placement",
        ),
    }
)


def _canonical_digest(value: Any) -> str:
    import json

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def rl_v09_field_rules_digest(profile: str = VERL_RL_V09_PROFILE) -> str:
    rules = _rules_for_profile(profile)
    return _canonical_digest(
        [
            {"field": field, "target": target, "classification": kind, "reason": reason}
            for field, (target, kind, reason) in sorted(rules.items())
        ]
    )


def _rules_for_profile(
    profile: str,
) -> dict[str, tuple[str | None, FieldClassification, str]]:
    if profile == VERL_RL_V09_PROFILE:
        return _RULES_V1
    if profile == VERL_RL_V09_PPO_PROFILE:
        return _RULES_V2
    raise ConfigError(f"unknown verl v0.9 RL compiler profile {profile!r}")


def _flatten(value: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, Mapping):
            flattened.update(_flatten(item, path))
        else:
            flattened[path] = item
    return flattened


_MISSING = object()


def _get(value: Mapping[str, Any], path: str, default: Any = _MISSING) -> Any:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            if default is _MISSING:
                raise ConfigError(f"resolved verl field {path} is required")
            return default
        current = current[part]
    return current


def _integer(value: Any, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise ConfigError(f"verl field {field} must be an integer >= {minimum}")
    if isinstance(value, str):
        try:
            parsed = int(value)
        except ValueError as exc:
            raise ConfigError(f"verl field {field} must be an integer >= {minimum}") from exc
        if str(parsed) != value.strip():
            raise ConfigError(f"verl field {field} must be an integer >= {minimum}")
    elif isinstance(value, int):
        parsed = value
    else:
        raise ConfigError(f"verl field {field} must be an integer >= {minimum}")
    if parsed < minimum:
        raise ConfigError(f"verl field {field} must be an integer >= {minimum}")
    return parsed


def _number(value: Any, field: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool):
        raise ConfigError(f"verl field {field} must be finite")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"verl field {field} must be finite") from exc
    if not math.isfinite(parsed) or (minimum is not None and parsed < minimum):
        raise ConfigError(f"verl field {field} must be finite and >= {minimum}")
    return parsed


def _bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"verl field {field} must be boolean")
    return value


def _list(value: Any, field: str) -> list[str]:
    values = [value] if isinstance(value, str) else value
    if not isinstance(values, list) or any(
        not isinstance(item, str) or not item for item in values
    ):
        raise ConfigError(f"verl field {field} must contain non-empty strings")
    return values


def _classify(
    flat: Mapping[str, Any],
    *,
    profile: str,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    rules = _rules_for_profile(profile)
    rows: list[dict[str, Any]] = []
    unsupported: list[str] = []
    unresolved: list[str] = []
    estimator = flat.get("algorithm.adv_estimator")
    estimator_specific = {
        "algorithm.gamma": "reinforce_plus_plus",
        "algorithm.lam": "gae",
        "algorithm.norm_adv_by_std_in_grpo": "grpo",
    }
    for field, value in sorted(flat.items()):
        rule = rules.get(field)
        if rule is None:
            target, kind, reason = None, "unsupported", "outside the versioned resolved subset"
            unsupported.append(field)
        else:
            target, kind, reason = rule
        if field == "actor_rollout_ref.actor.use_kl_loss" and value is False:
            kind = "exact"
            reason = "actor-loss KL is disabled in both runtimes"
        elif field == "actor_rollout_ref.actor.entropy_coeff":
            try:
                is_zero = float(value) == 0.0 and math.isfinite(float(value))
            except (TypeError, ValueError):
                is_zero = False
            if is_zero:
                kind = "exact"
                reason = "entropy regularization is disabled in both runtimes"
        elif field in estimator_specific and estimator != estimator_specific[field]:
            target = None
            kind = "informational_only"
            reason = f"not consumed by the {estimator!s} estimator"
        elif field == "algorithm.use_kl_in_reward" and value is False:
            kind = "exact"
            reason = "reference-policy reward KL is disabled in both runtimes"
        findings = audit_interpolation(value, label=field)
        if findings:
            unresolved.append(field)
        rows.append(
            {
                "upstream_field": field,
                "source_value": portable_payload(value),
                "local_target": target,
                "classification": kind,
                "reason": reason,
                "resolution_status": "unresolved" if findings else "resolved",
            }
        )
    return rows, unsupported, unresolved


def _algorithm(source: Mapping[str, Any], *, profile: str) -> str:
    estimator = str(_get(source, "algorithm.adv_estimator"))
    if estimator == "grpo":
        normalize = _bool(
            _get(source, "algorithm.norm_adv_by_std_in_grpo", True),
            "algorithm.norm_adv_by_std_in_grpo",
        )
        return "grpo" if normalize else "dr_grpo"
    if estimator in {"rloo", "reinforce_plus_plus"}:
        return estimator
    if estimator == "gae" and profile == VERL_RL_V09_PPO_PROFILE:
        return "ppo"
    reason = (
        "PPO/GAE needs a critic lifecycle that miniVERL does not yet implement"
        if estimator == "gae"
        else f"advantage estimator {estimator!r} is not implemented by this profile"
    )
    raise ConfigError(reason)


def _required_inputs(source: Mapping[str, Any], *, profile: str) -> list[dict[str, str]]:
    required: list[dict[str, str]] = []
    if _get(source, "trainer.total_training_steps", None) is None:
        required.append(
            {
                "field": "trainer.total_training_steps",
                "reason": "an exact rollout-iteration cap is required; epochs are dataset-dependent",
            }
        )
    if _get(source, "miniverl.reward.provider", None) is None:
        required.append(
            {
                "field": "miniverl.reward.provider",
                "reason": "the source must name a trusted local reward implementation",
            }
        )
    elif (
        profile == VERL_RL_V09_PPO_PROFILE
        and _get(source, "miniverl.reward.provider", None) == "hf_sequence_classifier"
    ):
        for field, reason in (
            ("reward_model.enable", "the trained reward role must be enabled explicitly"),
            ("reward_model.model.path", "the trained reward-model identity is required"),
            ("miniverl.reward.revision", "the reward model must use an immutable revision"),
        ):
            if _get(source, field, None) is None:
                required.append({"field": field, "reason": reason})
    if _get(source, "miniverl.actor.lora_enabled", None) is None:
        required.append(
            {
                "field": "miniverl.actor.lora_enabled",
                "reason": "PEFT versus full-parameter training is an explicit experiment choice",
            }
        )
    if (
        _get(source, "algorithm.use_kl_in_reward", False) is True
        or (
            profile == VERL_RL_V09_PPO_PROFILE
            and _get(source, "actor_rollout_ref.actor.use_kl_loss", False) is True
        )
    ) and _get(source, "miniverl.reference.adapter_path", None) is None:
        required.append(
            {
                "field": "miniverl.reference.adapter_path",
                "reason": "reference KL needs an explicit frozen reference-policy adapter",
            }
        )
    if (
        profile == VERL_RL_V09_PPO_PROFILE
        and _get(source, "algorithm.adv_estimator", None) == "gae"
    ):
        for field, reason in (
            ("critic.enable", "PPO requires an explicit trainable critic role"),
            ("critic.model.path", "PPO requires an explicit critic checkpoint identity"),
            (
                "miniverl.critic.lora_enabled",
                "the local critic parameterization must be chosen explicitly",
            ),
        ):
            if _get(source, field, None) is None:
                required.append({"field": field, "reason": reason})
    return required


def _recipe(
    source: Mapping[str, Any], *, profile: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    algorithm = _algorithm(source, profile=profile)
    use_kl_in_reward = _bool(
        _get(source, "algorithm.use_kl_in_reward", False), "algorithm.use_kl_in_reward"
    )
    actor_kl_enabled = _bool(
        _get(source, "actor_rollout_ref.actor.use_kl_loss", False),
        "actor_rollout_ref.actor.use_kl_loss",
    )
    if (
        not use_kl_in_reward
        and not actor_kl_enabled
        and _get(source, "miniverl.reference.adapter_path", None) is not None
    ):
        raise ConfigError("miniverl.reference is inactive unless algorithm.use_kl_in_reward=true")
    if actor_kl_enabled and profile != VERL_RL_V09_PPO_PROFILE:
        raise ConfigError("actor.use_kl_loss=true is not implemented by this compiler profile")
    entropy = _number(
        _get(source, "actor_rollout_ref.actor.entropy_coeff", 0.0),
        "actor_rollout_ref.actor.entropy_coeff",
        minimum=0.0,
    )
    if entropy != 0.0 and profile != VERL_RL_V09_PPO_PROFILE:
        raise ConfigError("nonzero actor entropy_coeff is not implemented by this compiler profile")
    ppo_epochs = _integer(
        _get(source, "actor_rollout_ref.actor.ppo_epochs", 1),
        "actor_rollout_ref.actor.ppo_epochs",
        minimum=1,
    )
    if ppo_epochs != 1 and profile != VERL_RL_V09_PPO_PROFILE:
        raise ConfigError("actor.ppo_epochs must be 1 in this compiler profile")
    aggregation = str(_get(source, "actor_rollout_ref.actor.loss_agg_mode", "token-mean"))
    if aggregation != "token-mean":
        raise ConfigError("only actor.loss_agg_mode=token-mean is semantically conformant locally")

    prompt_batch = _integer(
        _get(source, "data.train_batch_size"), "data.train_batch_size", minimum=1
    )
    n = _integer(_get(source, "actor_rollout_ref.rollout.n", 1), "rollout.n", minimum=1)
    if algorithm in {"grpo", "dr_grpo", "rloo"} and n < 2:
        raise ConfigError(f"algorithm {algorithm} requires actor_rollout_ref.rollout.n > 1")
    mini_batch = _integer(
        _get(source, "actor_rollout_ref.actor.ppo_mini_batch_size", prompt_batch),
        "actor_rollout_ref.actor.ppo_mini_batch_size",
        minimum=1,
    )
    trajectory_count = prompt_batch * n
    if mini_batch > trajectory_count:
        raise ConfigError(
            "actor.ppo_mini_batch_size cannot exceed data.train_batch_size * rollout.n"
        )
    prompt_length = _integer(
        _get(source, "data.max_prompt_length"), "data.max_prompt_length", minimum=1
    )
    response_length = _integer(
        _get(source, "data.max_response_length"), "data.max_response_length", minimum=1
    )
    provider = str(_get(source, "miniverl.reward.provider"))
    allowed_providers = {"exact_answer", "target_length"}
    if profile == VERL_RL_V09_PPO_PROFILE:
        allowed_providers.add("hf_sequence_classifier")
    if provider not in allowed_providers:
        raise ConfigError(
            "this portable profile accepts built-in exact_answer, target_length, and (in v2) "
            "hf_sequence_classifier rewards; Python reward objects must be injected explicitly "
            "by the local API"
        )
    if provider == "hf_sequence_classifier" and not _bool(
        _get(source, "reward_model.enable"), "reward_model.enable"
    ):
        raise ConfigError("hf_sequence_classifier requires reward_model.enable=true")
    local_backend = str(_get(source, "miniverl.rollout.backend", "hf_cached"))
    if local_backend not in {"hf_cached", "hf_reference"}:
        raise ConfigError("miniverl.rollout.backend must be hf_cached or hf_reference")
    source_engine = str(_get(source, "actor_rollout_ref.rollout.name", "unknown"))
    if local_backend == "hf_reference" and n > 1:
        raise ConfigError("grouped RL requires miniverl.rollout.backend=hf_cached")
    tensor_parallel = _integer(
        _get(source, "actor_rollout_ref.rollout.tensor_model_parallel_size", 1),
        "actor_rollout_ref.rollout.tensor_model_parallel_size",
        minimum=1,
    )
    gpus = _integer(
        _get(source, "trainer.n_gpus_per_node", 1), "trainer.n_gpus_per_node", minimum=1
    )
    nodes = _integer(_get(source, "trainer.nnodes", 1), "trainer.nnodes", minimum=1)
    lora_enabled = _bool(_get(source, "miniverl.actor.lora_enabled"), "miniverl.actor.lora_enabled")
    if (use_kl_in_reward or actor_kl_enabled) and not lora_enabled:
        raise ConfigError(
            "reference-KL temporal lowering requires miniverl.actor.lora_enabled=true"
        )
    if algorithm == "ppo":
        if not _bool(_get(source, "critic.enable"), "critic.enable"):
            raise ConfigError("PPO requires critic.enable=true")
        critic_path = str(_get(source, "critic.model.path"))
        actor_path = str(_get(source, "actor_rollout_ref.model.path"))
        if critic_path != actor_path:
            raise ConfigError(
                "the single-GPU PPO profile currently requires critic.model.path to equal "
                "actor_rollout_ref.model.path"
            )
        critic_lora = _bool(
            _get(source, "miniverl.critic.lora_enabled"),
            "miniverl.critic.lora_enabled",
        )
        if critic_lora != lora_enabled:
            raise ConfigError(
                "the local PPO profile requires actor and critic to use the same explicit "
                "LoRA parameterization"
            )
    kl_penalty = str(_get(source, "algorithm.kl_penalty", "kl"))
    if kl_penalty not in {"kl", "abs", "mse", "low_var_kl"}:
        raise ConfigError(f"algorithm.kl_penalty {kl_penalty!r} is not supported locally")
    kl_type = str(_get(source, "algorithm.kl_ctrl.type", "fixed"))
    if use_kl_in_reward and kl_type != "fixed":
        raise ConfigError("reference KL currently requires algorithm.kl_ctrl.type=fixed")
    kl_coef = (
        _number(
            _get(source, "algorithm.kl_ctrl.kl_coef"),
            "algorithm.kl_ctrl.kl_coef",
            minimum=0.0,
        )
        if use_kl_in_reward
        else 0.0
    )
    if use_kl_in_reward and kl_coef <= 0.0:
        raise ConfigError("reference KL requires a positive algorithm.kl_ctrl.kl_coef")
    target_modules = _list(
        _get(source, "actor_rollout_ref.model.target_modules", ["q_proj", "v_proj"]),
        "actor_rollout_ref.model.target_modules",
    )
    steps = _integer(
        _get(source, "trainer.total_training_steps"), "trainer.total_training_steps", minimum=1
    )
    save_freq = _integer(_get(source, "trainer.save_freq", -1), "trainer.save_freq", minimum=-1)
    test_freq = _integer(_get(source, "trainer.test_freq", -1), "trainer.test_freq", minimum=-1)
    recipe: dict[str, Any] = {
        "schema_version": 1,
        "run": {
            "name": str(_get(source, "trainer.experiment_name", "verl-local-rl")),
            "mode": "rl",
            "seed": _integer(_get(source, "data.seed", 0), "data.seed"),
            "output_dir": "runs",
            "deterministic": True,
            "tags": [
                str(_get(source, "trainer.project_name", "verl")),
                profile,
            ],
            "profile_identity": {
                "profile_name": profile,
                "upstream_tag": UPSTREAM_VERL_TAG,
                "upstream_commit": UPSTREAM_VERL_COMMIT,
                "field_rules_sha256": rl_v09_field_rules_digest(profile),
            },
        },
        "models": {
            "backend": "hf",
            "runtime": "shared_backbone"
            if (use_kl_in_reward or actor_kl_enabled)
            else "dual_model",
            "device": "cuda",
            "student": {
                "model_id": str(_get(source, "actor_rollout_ref.model.path")),
                "revision": _get(source, "miniverl.actor.revision", None),
                "tokenizer_revision": _get(
                    source,
                    "miniverl.actor.tokenizer_revision",
                    _get(source, "miniverl.actor.revision", None),
                ),
                "dtype": str(_get(source, "miniverl.actor.dtype", "auto")),
                "quantization": str(_get(source, "miniverl.actor.quantization", "none")),
                "gradient_checkpointing": _bool(
                    _get(source, "actor_rollout_ref.model.enable_gradient_checkpointing", False),
                    "actor_rollout_ref.model.enable_gradient_checkpointing",
                ),
                "lora": {
                    "enabled": lora_enabled,
                    "r": _integer(
                        _get(source, "actor_rollout_ref.model.lora_rank", 16),
                        "actor_rollout_ref.model.lora_rank",
                        minimum=1,
                    ),
                    "alpha": _integer(
                        _get(source, "actor_rollout_ref.model.lora_alpha", 32),
                        "actor_rollout_ref.model.lora_alpha",
                        minimum=1,
                    ),
                    "target_modules": target_modules,
                },
            },
        },
        "source": {
            "kind": "verl_parquet",
            "train_files": _list(_get(source, "data.train_files"), "data.train_files"),
            "val_files": _list(_get(source, "data.val_files", []), "data.val_files"),
            "prompt_key": str(_get(source, "data.prompt_key", "prompt")),
            "use_task_rewards": True,
            "max_prompt_length": prompt_length,
            "max_response_length": response_length,
            "truncation": str(_get(source, "data.truncation", "error")),
            "shuffle": _bool(_get(source, "data.shuffle", True), "data.shuffle"),
            "seed": _integer(_get(source, "data.seed", 0), "data.seed"),
        },
        "rollout": {
            "backend": local_backend,
            "samples_per_prompt": n,
            "prompt_batch_size": prompt_batch,
            "max_turns": 1,
            "max_new_tokens_per_turn": response_length,
            "max_total_tokens": prompt_length + response_length,
            "temperature": _number(
                _get(source, "actor_rollout_ref.rollout.temperature", 1.0),
                "actor_rollout_ref.rollout.temperature",
                minimum=0.0,
            ),
            "top_p": _number(
                _get(source, "actor_rollout_ref.rollout.top_p", 1.0),
                "actor_rollout_ref.rollout.top_p",
                minimum=0.0,
            ),
            "top_k": _integer(
                _get(source, "actor_rollout_ref.rollout.top_k", 0),
                "actor_rollout_ref.rollout.top_k",
            ),
            "record_logprobs": True,
        },
        "selection": {"selector": "all_model_tokens"},
        "loss": {
            "mode": "verl_rl_policy",
            "aggregation": "token-mean",
            "scale_by_temperature_squared": False,
            "clip_ratio": _number(
                _get(source, "actor_rollout_ref.actor.clip_ratio", 0.2),
                "actor_rollout_ref.actor.clip_ratio",
                minimum=0.0,
            ),
            "clip_ratio_low": _number(
                _get(source, "actor_rollout_ref.actor.clip_ratio_low", 0.2),
                "actor_rollout_ref.actor.clip_ratio_low",
                minimum=0.0,
            ),
            "clip_ratio_high": _number(
                _get(source, "actor_rollout_ref.actor.clip_ratio_high", 0.2),
                "actor_rollout_ref.actor.clip_ratio_high",
                minimum=0.0,
            ),
            "clip_ratio_c": _number(
                _get(source, "actor_rollout_ref.actor.clip_ratio_c", 3.0),
                "actor_rollout_ref.actor.clip_ratio_c",
                minimum=0.0,
            ),
        },
        "algorithm": {
            "name": algorithm,
            "implementation_version": ADVANTAGE_IMPLEMENTATION_VERSION,
            "gamma": _number(_get(source, "algorithm.gamma", 1.0), "algorithm.gamma", minimum=0.0),
            "lam": _number(_get(source, "algorithm.lam", 1.0), "algorithm.lam", minimum=0.0),
            "kl_coef": kl_coef,
            "kl_penalty": kl_penalty,
        },
        "reward": {"enabled": True, "provider": provider, "error_policy": "fail"},
        "train": {
            "cycles": steps,
            "rollouts_per_cycle": prompt_batch,
            "gradient_accumulation_steps": mini_batch,
            "trajectory_batch_size": _get(source, "miniverl.update.trajectory_batch_size", 1),
            "opd_freshness": (
                "strict" if algorithm == "ppo" or mini_batch == trajectory_count else "replay"
            ),
            "learning_rate": _number(
                _get(source, "actor_rollout_ref.actor.optim.lr"),
                "actor_rollout_ref.actor.optim.lr",
                minimum=0.0,
            ),
            "weight_decay": _number(
                _get(source, "actor_rollout_ref.actor.optim.weight_decay", 0.0),
                "actor_rollout_ref.actor.optim.weight_decay",
                minimum=0.0,
            ),
            "warmup_steps": _integer(
                _get(source, "actor_rollout_ref.actor.optim.lr_warmup_steps", 0),
                "actor_rollout_ref.actor.optim.lr_warmup_steps",
            ),
            "save_every_cycles": 0 if save_freq < 0 else save_freq,
            "eval_every_cycles": 0 if test_freq < 0 else test_freq,
        },
        "memory": {"strategy": str(_get(source, "miniverl.memory.strategy", "auto"))},
        "eval": {"enabled": False},
    }
    if profile == VERL_RL_V09_PPO_PROFILE:
        recipe["algorithm"].update(
            {
                "actor_kl_coef": (
                    _number(
                        _get(source, "actor_rollout_ref.actor.kl_loss_coef", 0.001),
                        "actor_rollout_ref.actor.kl_loss_coef",
                        minimum=0.0,
                    )
                    if actor_kl_enabled
                    else 0.0
                ),
                "actor_kl_penalty": str(_get(source, "actor_rollout_ref.actor.kl_loss_type", "kl")),
                "entropy_coeff": entropy,
                "actor_ppo_epochs": ppo_epochs,
            }
        )
    if provider == "hf_sequence_classifier":
        tokenizer_revision = _get(source, "miniverl.reward.tokenizer_revision", None)
        recipe["reward"]["model"] = {
            "model_id": str(_get(source, "reward_model.model.path")),
            "revision": str(_get(source, "miniverl.reward.revision")),
            "tokenizer_revision": tokenizer_revision,
            "positive_class_index": _integer(
                _get(source, "miniverl.reward.positive_class_index", 1),
                "miniverl.reward.positive_class_index",
                minimum=0,
            ),
            "max_length": _integer(
                _get(source, "miniverl.reward.max_length", 512),
                "miniverl.reward.max_length",
                minimum=8,
            ),
            "batch_size": _integer(
                _get(source, "miniverl.reward.batch_size", 4),
                "miniverl.reward.batch_size",
                minimum=1,
            ),
            "timeout_seconds": _number(
                _get(source, "miniverl.reward.timeout_seconds", 120.0),
                "miniverl.reward.timeout_seconds",
                minimum=0.000001,
            ),
            "trust_remote_code": _bool(
                _get(source, "reward_model.model.trust_remote_code", False),
                "reward_model.model.trust_remote_code",
            ),
            "offload_between_phases": _bool(
                _get(source, "miniverl.reward.offload_between_phases", True),
                "miniverl.reward.offload_between_phases",
            ),
        }
    if algorithm == "ppo":
        critic_mini_batch = _integer(
            _get(source, "critic.ppo_mini_batch_size", mini_batch),
            "critic.ppo_mini_batch_size",
            minimum=1,
        )
        if critic_mini_batch != mini_batch:
            raise ConfigError(
                "the local PPO profile currently requires critic.ppo_mini_batch_size to "
                "equal actor_rollout_ref.actor.ppo_mini_batch_size"
            )
        recipe["critic"] = {
            "enabled": True,
            "learning_rate": _number(
                _get(source, "critic.optim.lr"), "critic.optim.lr", minimum=0.0
            ),
            "weight_decay": _number(
                _get(source, "critic.optim.weight_decay", 0.0),
                "critic.optim.weight_decay",
                minimum=0.0,
            ),
            "warmup_steps": _integer(
                _get(source, "critic.optim.lr_warmup_steps", 0),
                "critic.optim.lr_warmup_steps",
            ),
            "ppo_epochs": _integer(
                _get(source, "critic.ppo_epochs", ppo_epochs),
                "critic.ppo_epochs",
                minimum=1,
            ),
        }
        recipe["algorithm"]["cliprange_value"] = _number(
            _get(source, "critic.cliprange_value", 0.5),
            "critic.cliprange_value",
            minimum=0.0,
        )
    if use_kl_in_reward or actor_kl_enabled:
        adapter_source = str(_get(source, "miniverl.reference.adapter_source", "local"))
        adapter: dict[str, Any] = {
            "path": str(_get(source, "miniverl.reference.adapter_path")),
            "source": adapter_source,
        }
        adapter_revision = _get(source, "miniverl.reference.adapter_revision", None)
        if adapter_revision is not None:
            adapter["revision"] = str(adapter_revision)
        recipe["models"]["reference"] = {
            "model_id": str(_get(source, "actor_rollout_ref.model.path")),
            "revision": _get(source, "miniverl.actor.revision", None),
            "tokenizer_revision": _get(
                source,
                "miniverl.actor.tokenizer_revision",
                _get(source, "miniverl.actor.revision", None),
            ),
            "dtype": str(_get(source, "miniverl.actor.dtype", "auto")),
            "quantization": str(_get(source, "miniverl.actor.quantization", "none")),
            "adapter": adapter,
        }
    lowering: list[dict[str, Any]] = [
        {
            "source_field": "actor_rollout_ref.rollout.name",
            "source_value": source_engine,
            "local_value": local_backend,
            "kind": "physical_runtime",
        },
        {
            "source_field": "actor_rollout_ref.rollout.tensor_model_parallel_size",
            "source_value": tensor_parallel,
            "local_value": 1,
            "kind": "distributed_only",
        },
        {
            "source_field": "trainer.n_gpus_per_node/trainer.nnodes",
            "source_value": {"n_gpus_per_node": gpus, "nnodes": nodes},
            "local_value": {"devices": 1, "processes": 1},
            "kind": "distributed_only",
        },
    ]
    return recipe, lowering


def publish_imported_verl_rl_v09(
    source: str | Path,
    *,
    out: str | Path,
    target_verl: str,
    overwrite: bool = False,
    lock_timeout: float = DEFAULT_LOCK_TIMEOUT,
    profile: str = VERL_RL_V09_PROFILE,
) -> dict[str, Any]:
    """Publish a native recipe, a user-input template, or a rejection report."""
    if target_verl not in {UPSTREAM_VERL_TAG, UPSTREAM_VERL_COMMIT}:
        raise ConfigError(
            f"profile {profile} targets {UPSTREAM_VERL_TAG} or "
            f"{UPSTREAM_VERL_COMMIT}, not {target_verl!r}"
        )
    source_path = Path(source)
    targets = import_output_targets(out)
    reject_source_output_alias({"source config": source_path}, targets)
    try:
        source_bytes = source_path.read_bytes()
        payload = yaml.safe_load(source_bytes)
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot read verl config {source_path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ConfigError("verl config must contain one YAML mapping")
    _rules_for_profile(profile)
    flat = _flatten(payload)
    fields, unsupported, unresolved = _classify(flat, profile=profile)
    required = _required_inputs(payload, profile=profile)
    common: dict[str, Any] = {
        "schema_version": 1,
        "profile": profile,
        "profile_field_rules_sha256": rl_v09_field_rules_digest(profile),
        "source_verl": {
            "repository": VERL_REPOSITORY,
            "tag": UPSTREAM_VERL_TAG,
            "commit": UPSTREAM_VERL_COMMIT,
        },
        "source_config_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "input_contract": "resolved documented verl v0.9 RL subset; not arbitrary Hydra YAML",
        "field_classification": fields,
        "unsupported_fields": unsupported,
        "unresolved_fields": unresolved,
        "required_user_input": required,
        "report_path": targets["report"].name,
    }
    transaction = OutputTransaction(
        targets=targets,
        stem=targets["recipe"].stem,
        lock_root=targets["recipe"].parent,
        overwrite=overwrite,
        lock_timeout=lock_timeout,
    ).begin()
    try:
        if unsupported or unresolved:
            reason = "unsupported fields" if unsupported else "unresolved interpolation"
            report = {
                **common,
                "status": "rejected",
                "executable": False,
                "reason": reason,
                "generated_path": targets["report"].name,
            }
            transaction.write_json("report", report)
            transaction.discard("recipe")
            transaction.discard("template")
            transaction.commit()
            return report
        if required:
            template = {
                "schema_version": 1,
                "status": "needs_user_input",
                "note": "Non-executable source template; supply every required field.",
                "profile": profile,
                "source_values": portable_payload(dict(payload)),
                "required_user_input": required,
            }
            report = {
                **common,
                "status": "needs_user_input",
                "executable": False,
                "generated_path": targets["template"].name,
            }
            transaction.write_bytes(
                "template",
                yaml.safe_dump(template, sort_keys=False, allow_unicode=True).encode("utf-8"),
            )
            transaction.write_json("report", report)
            transaction.discard("recipe")
            transaction.commit()
            return report
        try:
            recipe, lowering = _recipe(payload, profile=profile)
            from miniverl.config import RunConfig

            validated = RunConfig.from_mapping(recipe)
        except ConfigError:
            raise
        except Exception as exc:
            raise ConfigError(f"generated local RL recipe failed validation: {exc}") from exc
        rendered = yaml.safe_dump(recipe, sort_keys=False, allow_unicode=True, width=100).encode()
        report = {
            **common,
            "status": "accepted",
            "executable": True,
            "generated_path": targets["recipe"].name,
            "generated_recipe_sha256": hashlib.sha256(rendered).hexdigest(),
            "generated_recipe_validated": True,
            "algorithm": validated.algorithm.name.value,
            "local_lowering": lowering,
        }
        transaction.write_bytes("recipe", rendered)
        transaction.write_json("report", report)
        transaction.discard("template")
        transaction.commit()
        return report
    finally:
        transaction.close()
