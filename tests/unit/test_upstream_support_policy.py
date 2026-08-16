from __future__ import annotations

import json
from pathlib import Path


def test_support_matrix_matches_the_closed_profile_registry() -> None:
    from miniverl.bridge.profiles import get_profile, list_profiles

    matrix = json.loads(
        Path("docs/generated/upstream-support-matrix.json").read_text(encoding="utf-8")
    )
    assert matrix["schema_version"] == 1
    assert matrix["policy"] == "closed_immutable_profiles"
    assert matrix["unprofiled_versions"] == "unsupported"
    assert [row["name"] for row in matrix["profiles"]] == [
        profile.name for profile in list_profiles()
    ]
    for row in matrix["profiles"]:
        identity = get_profile(row["name"]).identity
        assert row["status"] == "active"
        assert row["upstream_tag"] == identity.upstream_tag
        assert row["upstream_commit"] == identity.upstream_commit
        assert row["distributed_execution_tested"] is False
