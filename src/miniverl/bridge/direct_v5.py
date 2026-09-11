"""Native-Hydra local RL profile; v4 rules and identity remain immutable."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from miniverl.bridge import direct as v4
from miniverl.errors import ConfigError

PROFILE = "verl-rl-v0.9-single-gpu-v5"


class SamplingSemantics(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    actor_shuffle: bool = False
    actor_seed: int = Field(default=42, ge=0, lt=2**63)
    critic_shuffle: bool = False
    critic_seed: int = Field(default=42, ge=0, lt=2**63)
    filter_metric: str | None = None
    refill_failed_groups: bool = False
    max_inflight_gen_batches: int = Field(default=1, ge=1, le=1024)
    max_attempted_groups: int | None = Field(default=None, ge=1)


def settings(source: dict[str, Any]) -> SamplingSemantics:
    def get(key: str, default: Any = None) -> Any:
        return v4._get(source, key, default)

    enabled = get("algorithm.filter_groups.enable", False)
    if type(enabled) is not bool:
        raise ConfigError("algorithm.filter_groups.enable must be boolean")
    metric = get("algorithm.filter_groups.metric") if enabled else None
    if enabled and (not isinstance(metric, str) or not metric):
        raise ConfigError("group filtering requires a named reward_extra_info metric, e.g. score")
    if (
        enabled
        and get("reward.reward_model.enable", False)
        and not get("reward.reward_model.enable_resource_pool", False)
    ):
        raise ConfigError("upstream group filtering requires rewards available before sampling")
    return SamplingSemantics(
        actor_shuffle=get("actor_rollout_ref.actor.shuffle", False),
        actor_seed=get("actor_rollout_ref.actor.data_loader_seed", 42),
        critic_shuffle=get("critic.shuffle", False)
        if get("algorithm.adv_estimator") == "gae"
        else False,
        critic_seed=get("critic.data_loader_seed", 42)
        if get("algorithm.adv_estimator") == "gae"
        else 42,
        filter_metric=metric,
        refill_failed_groups=get("trainer.v1.sampler.sync_refill_failed_groups", False),
        max_inflight_gen_batches=get("algorithm.filter_groups.max_inflight_gen_batches", 1),
        max_attempted_groups=get("miniverl.v5.max_attempted_groups"),
    )


def rules_digest() -> str:
    return v4._digest(
        {
            "v4_rules": v4.direct_rules_digest(),
            "implementation": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "sampling_schema": SamplingSemantics.model_json_schema(),
        }
    )


def build_native(source: dict[str, Any], *, cycles: int) -> dict[str, Any]:
    recipe = v4.build_native(source, cycles=cycles)
    semantic = settings(source)
    recipe["run"]["profile_identity"].update(
        {
            "profile_name": PROFILE,
            "field_rules_sha256": rules_digest(),
            "v5_sampling": semantic.model_dump(mode="json"),
        }
    )
    recipe["run"]["tags"] = [source.get("trainer", {}).get("project_name", "verl"), PROFILE]
    return recipe


def compile_direct(source: str | Path, *, bindings: list[str] | None = None) -> dict[str, Any]:
    ordinary = []
    attempt_limit = None
    for assignment in bindings or []:
        key, separator, value = assignment.partition("=")
        if key != "sampling.max_attempted_groups":
            ordinary.append(assignment)
        elif not separator or not value.isdecimal() or int(value) < 1 or attempt_limit is not None:
            raise ConfigError("sampling.max_attempted_groups requires one positive integer binding")
        else:
            attempt_limit = int(value)
    report = v4.compile_direct(source, bindings=ordinary)
    effective = report["effective_source"]
    if attempt_limit is not None:
        v4._put(effective, "miniverl.v5.max_attempted_groups", attempt_limit)
        report["runtime_bindings"].append(
            {
                "binding": "sampling.max_attempted_groups",
                "bound": attempt_limit,
                "target": "local execution guard",
                "original": None,
            }
        )
    replacements = {
        "actor_rollout_ref.actor.shuffle": (
            "exact",
            "upstream DataLoader ordering across PPO epochs",
        ),
        "critic.shuffle": ("exact", "independent upstream critic DataLoader ordering"),
        "actor_rollout_ref.actor.data_loader_seed": ("exact", "actor-local torch.Generator seed"),
        "critic.data_loader_seed": ("exact", "critic-local torch.Generator seed"),
        "algorithm.filter_groups.enable": (
            "locally_lowered",
            "upstream sync group-metric filter; deterministic dispatch-order tie-break for equal-age surplus",
        ),
        "algorithm.filter_groups.metric": ("exact", "reward_extra_info group metric"),
        "algorithm.filter_groups.max_num_gen_batches": (
            "informational_only",
            "pinned V1 replay buffer ignores this legacy retry bound",
        ),
        "algorithm.filter_groups.max_inflight_gen_batches": (
            "locally_lowered",
            "bounded synchronous dispatch followed by drain and surplus discard",
        ),
        "trainer.v1.sampler.sync_refill_failed_groups": ("exact", "replace fully failed groups"),
        "actor_rollout_ref.rollout.over_sample_rate": (
            "informational_only",
            "legacy declaration with no execution consumer in pinned verl v0.9.0",
        ),
    }
    for row in report["fields"]:
        if row["field"] in replacements:
            row["classification"], row["reason"] = replacements[row["field"]]
    report["rejections"] = [r for r in report["rejections"] if r["field"] not in replacements]
    try:
        semantic = settings(effective)
        report["sampling_semantics"] = semantic.model_dump(mode="json")
        if semantic.filter_metric or semantic.refill_failed_groups:
            for row in report["fields"]:
                if row["field"] == "data.gen_batch_size":
                    row.update(
                        classification="informational_only",
                        reason="pinned synchronous V1 sampler forces one prompt per generation dispatch when filtering or refilling",
                    )
            report["rejections"] = [
                r for r in report["rejections"] if r["field"] != "data.gen_batch_size"
            ]
    except (ValueError, ConfigError) as exc:
        report["rejections"].append({"field": "sampling", "reason": str(exc)})
        report["ir_schema_validation_passed"] = False
    report.update({"profile": PROFILE, "field_rules_sha256": rules_digest()})
    report["semantic_status"] = "rejected" if report["rejections"] else "accepted"
    report["execution_status"] = (
        "unsupported_semantics"
        if report["rejections"]
        else "requires_bindings"
        if report["required_bindings"]
        else "preflight_required"
    )
    return report
