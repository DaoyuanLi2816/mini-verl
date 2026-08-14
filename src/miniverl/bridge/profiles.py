"""Closed, versioned registry for miniVERL compatibility profiles.

The registry is deliberately internal and static.  It never imports entry
points or executes third-party profile code.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from miniverl.bridge.contract import VERL_COMMIT, VERL_REPOSITORY, VERL_TAG
from miniverl.bridge.opd_v08 import (
    VERL_OPD_V08_PROFILE,
    VerlOPDV08Profile,
    compatibility_rule,
    field_rules_digest,
    load_verl_opd_v08_source,
)
from miniverl.errors import ConfigError
from miniverl.utils.runs import canonical_json

__all__ = [
    "CompatibilityCheck",
    "CompatibilityExplanation",
    "CompatibilityProfile",
    "ProfileIdentity",
    "ProfileSummary",
    "check_profile",
    "get_profile",
    "list_profiles",
]


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ProfileIdentity(_FrozenModel):
    """All version axes that may change profile behavior or portable bytes."""

    profile_name: str
    profile_schema_version: int = Field(ge=1)
    upstream_repository: str
    upstream_tag: str
    upstream_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    field_rule_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    native_compiler_version: str
    loss_conformance_version: str
    export_version: str
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(cls, **values: Any) -> ProfileIdentity:
        payload = {key: value for key, value in values.items() if key != "digest"}
        digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
        return cls(**payload, digest=digest)


class ProfileSummary(_FrozenModel):
    name: str
    objective: str
    teacher_target: str
    status: Literal["measured", "conformance_only", "experimental"]
    identity: ProfileIdentity


class CompatibilityExplanation(_FrozenModel):
    profile: str
    upstream_field: str
    classification: str
    local_target: str | None
    reason: str
    semantic_risk: str
    user_confirmation_required: bool
    supported_algorithm: bool
    field_accepted: bool
    field_effective: bool
    field_locally_reinterpreted: bool
    field_informational: bool
    field_unsupported: bool
    profile_applicable: bool


class CompatibilityCheck(_FrozenModel):
    schema_version: Literal[1] = 1
    status: Literal["compatible", "needs_user_confirmation", "unsupported"]
    executable: bool
    profile_identity: ProfileIdentity
    source_digest: str
    compiled_digest: str
    summary: dict[str, int | bool]
    fields: list[dict[str, Any]]


class CompatibilityProfile(ABC):
    """Typed operations implemented by each built-in, immutable profile."""

    @property
    @abstractmethod
    def identity(self) -> ProfileIdentity: ...

    @property
    @abstractmethod
    def summary(self) -> ProfileSummary: ...

    @abstractmethod
    def config_schema(self) -> dict[str, Any]: ...

    @abstractmethod
    def explain(self, field: str) -> CompatibilityExplanation: ...

    @abstractmethod
    def check(
        self, source: str | Path, *, accept_local_reinterpretations: bool = False
    ) -> CompatibilityCheck: ...

    @abstractmethod
    def show(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class _DirectGKDProfile(CompatibilityProfile):
    @property
    def identity(self) -> ProfileIdentity:
        return ProfileIdentity.create(
            profile_name=VERL_OPD_V08_PROFILE,
            profile_schema_version=1,
            upstream_repository=VERL_REPOSITORY,
            upstream_tag=VERL_TAG,
            upstream_commit=VERL_COMMIT,
            field_rule_digest=field_rules_digest(),
            native_compiler_version="direct-gkd-native-v1",
            loss_conformance_version="forward-kl-topk-verl-v0.8-v1",
            export_version="verl-opd-export-v1",
        )

    @property
    def summary(self) -> ProfileSummary:
        return ProfileSummary(
            name=VERL_OPD_V08_PROFILE,
            objective="direct GKD forward_kl_topk",
            teacher_target="top-k token IDs and log-probabilities",
            status="measured",
            identity=self.identity,
        )

    def config_schema(self) -> dict[str, Any]:
        return VerlOPDV08Profile.model_json_schema()

    def explain(self, field: str) -> CompatibilityExplanation:
        rule = compatibility_rule(field)
        classification = str(rule["classification"])
        unsupported = classification == "unsupported"
        informational = classification == "informational_only"
        return CompatibilityExplanation(
            profile=VERL_OPD_V08_PROFILE,
            upstream_field=field,
            classification=classification,
            local_target=rule["local_target"],
            reason=str(rule["reason"]),
            semantic_risk=str(rule["semantic_risk"]),
            user_confirmation_required=bool(rule["user_confirmation_required"]),
            supported_algorithm=True,
            field_accepted=not unsupported,
            field_effective=not unsupported
            and not informational
            and rule["local_target"] is not None,
            field_locally_reinterpreted=classification == "locally_reinterpreted",
            field_informational=informational,
            field_unsupported=unsupported,
            profile_applicable=True,
        )

    def check(
        self, source: str | Path, *, accept_local_reinterpretations: bool = False
    ) -> CompatibilityCheck:
        compiled = load_verl_opd_v08_source(
            source,
            accept_local_reinterpretations=accept_local_reinterpretations,
            require_executable=False,
        )
        fields = [item.model_dump(mode="json") for item in compiled.compatibility]
        unsupported = sum(not item.executable for item in compiled.compatibility)
        informational = sum(
            item.classification == "informational_only" for item in compiled.compatibility
        )
        reinterpreted = sum(
            item.classification == "locally_reinterpreted" for item in compiled.compatibility
        )
        effective = sum(
            item.executable
            and item.classification != "informational_only"
            and item.local_target is not None
            for item in compiled.compatibility
        )
        accepted = bool(compiled.reinterpretation_acceptance["accepted"])
        if unsupported:
            status: Literal["compatible", "needs_user_confirmation", "unsupported"] = "unsupported"
        elif not accepted:
            status = "needs_user_confirmation"
        else:
            status = "compatible"
        return CompatibilityCheck(
            status=status,
            executable=compiled.executable and accepted,
            profile_identity=self.identity,
            source_digest=compiled.source_digest,
            compiled_digest=compiled.compiled_digest,
            summary={
                "supported_algorithm": True,
                "field_accepted": len(fields) - unsupported,
                "field_effective": effective,
                "field_locally_reinterpreted": reinterpreted,
                "field_informational": informational,
                "field_unsupported": unsupported,
                "profile_not_applicable": False,
            },
            fields=fields,
        )

    def show(self) -> dict[str, Any]:
        from importlib.resources import files

        minimal_yaml = (
            files("miniverl")
            .joinpath("resources/qwen3_0_6b_1_7b_opd.yaml")
            .read_text(encoding="utf-8")
        )
        return {
            **self.summary.model_dump(mode="json"),
            "identity": self.identity.model_dump(mode="json"),
            "algorithm_contract": {
                "actor_count": 1,
                "teacher_count": 1,
                "generations_per_prompt": 1,
                "objective": "forward_kl_topk",
                "task_rewards": False,
                "distributed_execution": False,
                "device": "one local CUDA GPU",
            },
            "minimal_yaml": minimal_yaml,
            "override_invocation": (
                "miniverl plan --profile verl-opd-v0.8-single-gpu-v1 "
                "--config builtin:qwen3-0.6b-1.7b-opd "
                "--set data.train_batch_size=4 --json"
            ),
        }


_REGISTRY: dict[str, CompatibilityProfile] = {
    VERL_OPD_V08_PROFILE: _DirectGKDProfile(),
}


def list_profiles() -> tuple[ProfileSummary, ...]:
    return tuple(_REGISTRY[name].summary for name in sorted(_REGISTRY))


def get_profile(name: str) -> CompatibilityProfile:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise ConfigError(
            f"unknown compatibility profile {name!r}",
            hint="run 'miniverl profiles list' for the closed built-in registry",
        ) from exc


def check_profile(
    name: str,
    source: str | Path,
    *,
    accept_local_reinterpretations: bool = False,
) -> CompatibilityCheck:
    return get_profile(name).check(
        source, accept_local_reinterpretations=accept_local_reinterpretations
    )
