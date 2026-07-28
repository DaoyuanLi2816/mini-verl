"""Opt-in checks for the immutable public protocol-teacher artifact."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.network

_REPO_ID = "DaoyuanLi/mini-verl-qwen3-1.7b-protocol-teacher"
_REVISION = "23323751318135484c06c043b1f9b9e7016dd89f"
_BASE_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
_TOKENIZER_FINGERPRINT = "f2f5e826dddc3ff1e2481111075f2ed6eced4e553168222d67650931d25be035"
_WEIGHTS_SHA256 = "8df7e7bc1b8283b910aa13bc4173083ae20c838bcacb366d7dbcabc7b310b994"
_MANIFEST_SHA256 = "502bca7489c6fe161ebf198d2a1b4622123d4f958885a7e4714c6a02a2e1ac43"


def test_protocol_teacher_hub_revision_has_verified_miniverl_provenance() -> None:
    from miniverl.config.models import (
        AdapterSource,
        TeacherAdapterConfig,
        TeacherModelConfig,
    )
    from miniverl.models.adapter_io import validate_teacher_adapter

    provenance = validate_teacher_adapter(
        TeacherAdapterConfig(
            source=AdapterSource.HUB,
            path=_REPO_ID,
            revision=_REVISION,
            require_policy_evaluation=True,
            minimum_strict_success_rate=0.5,
        ),
        TeacherModelConfig(
            model_id="Qwen/Qwen3-1.7B",
            revision=_BASE_REVISION,
            tokenizer_revision=_BASE_REVISION,
        ),
        tokenizer_fingerprint=_TOKENIZER_FINGERPRINT,
    )

    assert provenance["revision"] == _REVISION
    assert provenance["weights_sha256"] == _WEIGHTS_SHA256
    assert provenance["manifest_digest"] == _MANIFEST_SHA256
    assert provenance["policy_evaluation"]["strict_task_success_rate"] == 1.0
