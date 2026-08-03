from __future__ import annotations

import hashlib
import json
from pathlib import Path


def test_community_template_is_schema_valid_private_and_hash_bound(tmp_path: Path) -> None:
    from miniverl.bridge.community import export_community_submission, validate_submission

    out = tmp_path / "submission.json"
    payload = export_community_submission(out)
    assert payload == json.loads(out.read_text(encoding="utf-8"))
    assert payload["measured_status"] == "not_measured"
    assert payload["recipe_sha256"]
    assert payload["compatible_verl_bridge_profile"] == ("single-gpu-online-distillation-v1")
    serialized = out.read_text(encoding="utf-8").lower()
    assert "hostname" not in serialized
    assert "username" not in serialized
    assert "c:\\users" not in serialized
    assert validate_submission(out) == []


def test_community_validator_rejects_unverified_hashes_and_secrets(tmp_path: Path) -> None:
    from miniverl.bridge.community import export_community_submission, validate_submission

    out = tmp_path / "submission.json"
    payload = export_community_submission(out)
    payload["recipe_sha256"] = "0" * 64
    payload["notes"] = "token=hf_abcdefghijklmnopqrstuvwxyz123456"
    out.write_text(json.dumps(payload), encoding="utf-8")
    problems = validate_submission(out)
    assert any("recipe_sha256" in problem for problem in problems)
    assert any("privacy" in problem for problem in problems)


def test_packaged_recipe_registry_has_five_honest_hash_bound_categories() -> None:
    from miniverl.bridge.community import load_recipe_registry

    root = Path(__file__).resolve().parents[2]
    records = load_recipe_registry()
    assert {record["category"] for record in records} == {
        "policy-conditioned alignment",
        "preference-teacher distillation",
        "localized safety/policy OPD",
        "tool-policy alignment",
        "RecoveryBench",
    }
    assert sum(record["measured_status"] == "measured" for record in records) == 4
    for record in records:
        assert record["model"]["revision"]
        assert record["hardware"]["gpu"]
        assert record["benchmark"]
        if record["measured_status"] == "measured":
            assert record["hardware"]["vram_gib"]
            assert record["wall_time_seconds"] is not None
        else:
            assert record["recipe_id"] == "preference-teacher-distillation"
            assert record["hardware"]["vram_gib"] is None
            assert record["wall_time_seconds"] is None
        assert record["compatible_miniverl_release"] == "0.6.0"
        assert record["compatible_verl_bridge_profile"] == ("single-gpu-online-distillation-v1")
        artifact = root / record["artifact"]["path"]
        assert artifact.is_file()
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == record["artifact"]["sha256"]
