from __future__ import annotations

from pathlib import Path

import pytest

from miniverl.errors import ConfigError

DIRECT_PROFILE = "verl-opd-v0.8-single-gpu-v1"
PG_PROFILE = "verl-opd-v0.8-single-gpu-pg-k1-v1"


def test_registry_exposes_immutable_direct_gkd_identity() -> None:
    from miniverl.bridge.profiles import get_profile, list_profiles

    profiles = list_profiles()
    assert [item.name for item in profiles] == [DIRECT_PROFILE, PG_PROFILE]

    profile = get_profile(DIRECT_PROFILE)
    identity = profile.identity
    assert identity.profile_name == DIRECT_PROFILE
    assert identity.profile_schema_version == 1
    assert identity.upstream_tag == "v0.8.0"
    assert identity.upstream_commit == "7aed6b230776f963fa09509c10d9c3a767d1102c"
    assert len(identity.field_rule_digest) == 64
    assert len(identity.digest) == 64
    assert identity.digest == profile.identity.digest


def test_registry_profile_identity_changes_when_contract_version_changes() -> None:
    from miniverl.bridge.profiles import ProfileIdentity

    base = ProfileIdentity.create(
        profile_name=DIRECT_PROFILE,
        profile_schema_version=1,
        upstream_repository="https://github.com/verl-project/verl",
        upstream_tag="v0.8.0",
        upstream_commit="7aed6b230776f963fa09509c10d9c3a767d1102c",
        field_rule_digest="a" * 64,
        native_compiler_version="direct-gkd-native-v1",
        loss_conformance_version="forward-kl-topk-verl-v0.8-v1",
        export_version="verl-opd-export-v1",
    )
    changed = ProfileIdentity.create(
        **{
            **base.model_dump(mode="python", exclude={"digest"}),
            "native_compiler_version": "direct-gkd-native-v2",
        }
    )
    assert base.digest != changed.digest


def test_profile_schema_and_field_explanation_are_typed() -> None:
    from miniverl.bridge.profiles import get_profile

    profile = get_profile(DIRECT_PROFILE)
    schema = profile.config_schema()
    assert schema["title"] == "VerlOPDV08Profile"
    assert schema["additionalProperties"] is False

    explanation = profile.explain("actor_rollout_ref.actor.ppo_mini_batch_size")
    assert explanation.classification == "informational_only"
    assert explanation.local_target is None
    assert explanation.field_accepted is True
    assert explanation.field_effective is False
    assert "direct" in explanation.reason.lower()

    with pytest.raises(ConfigError, match="not part of profile"):
        profile.explain("critic.model.path")


def test_profile_check_compiles_and_reports_effective_field_classes() -> None:
    from miniverl.bridge.profiles import check_profile

    config = Path(__file__).resolve().parents[2] / "examples/verl-opd-v0.8-single-gpu.yaml"
    report = check_profile(DIRECT_PROFILE, config, accept_local_reinterpretations=True)
    assert report.status == "compatible"
    assert report.executable is True
    assert report.profile_identity.profile_name == DIRECT_PROFILE
    assert report.summary["supported_algorithm"] is True
    assert report.summary["field_accepted"] > 0
    assert report.summary["field_effective"] > 0
    assert report.summary["field_locally_reinterpreted"] > 0
    assert report.summary["field_informational"] > 0
    assert report.summary["field_unsupported"] == 0


def test_unknown_profile_fails_closed_without_plugin_loading() -> None:
    from miniverl.bridge.profiles import get_profile

    with pytest.raises(ConfigError, match="unknown compatibility profile"):
        get_profile("third-party:anything")
