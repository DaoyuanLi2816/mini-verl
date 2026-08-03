from __future__ import annotations

import json
from pathlib import Path


def test_smoke_source_builds_a_doctor_clean_bundle(tmp_path: Path) -> None:
    from miniverl.bridge.contract import VERL_TAG
    from miniverl.bridge.doctor import inspect_bridge_bundle
    from miniverl.bridge.export import export_verl_bundle
    from scripts.prepare_verl_bridge_smoke import prepare_smoke_run

    source = prepare_smoke_run(tmp_path / "source")
    bundle = tmp_path / "bundle"
    export_verl_bundle(source, target_verl=VERL_TAG, out=bundle)
    diagnosis = inspect_bridge_bundle(bundle)
    assert diagnosis["verdict"] == "ok"
    assert diagnosis["installed_verl"]["status"] in {"not installed", "unverified", "ok"}
    assert diagnosis["distributed_execution_status"] == "not tested"


def test_committed_exact_source_smoke_is_pin_bound_and_does_not_claim_scale_out() -> None:
    from miniverl.bridge.contract import BRIDGE_PROFILE, VERL_COMMIT, VERL_TAG

    root = Path(__file__).resolve().parents[2]
    record = json.loads(
        (root / "docs" / "generated" / "verl-bridge-smoke.json").read_text(encoding="utf-8")
    )
    assert record["verdict"] == "ok"
    assert record["profile"] == BRIDGE_PROFILE
    assert record["target_verl"] == {
        "commit": VERL_COMMIT,
        "observed_package_version": "0.8.0.dev0",
        "observed_vcs_commit": VERL_COMMIT,
        "tag": VERL_TAG,
    }
    assert record["official_config"]["missing_fields"] == []
    assert len(record["official_config"]["required_fields"]) == 14
    assert record["official_config"]["missing_export_fields"] == []
    assert record["official_config"]["structured_merge_status"] == "passed"
    assert record["bundle_doctor"]["verdict"] == "ok"
    assert record["bundle_doctor"]["artifact_hashes"]["problems"] == []
    assert record["bundle_doctor"]["artifact_hashes"]["files"] == 14
    assert record["distributed_execution_status"] == "not tested"
    assert record["tiny_cpu_dry_run"]["status"] == "artifact-only"
