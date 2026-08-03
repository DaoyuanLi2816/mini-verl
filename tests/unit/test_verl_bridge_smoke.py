from __future__ import annotations

import importlib.util
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


def test_bridge_diagrams_are_generated_responsive_and_fail_closed() -> None:
    root = Path(__file__).resolve().parents[2]
    script = root / "scripts" / "publish_verl_bridge_diagrams.py"
    spec = importlib.util.spec_from_file_location("publish_verl_bridge_diagrams", script)
    assert spec and spec.loader
    publisher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(publisher)

    rendered = publisher.render_diagrams()
    assert set(rendered) == {
        "verl-bridge-architecture.svg",
        "verl-bridge-architecture-mobile.svg",
    }
    for name, content in rendered.items():
        assert (root / "docs" / name).read_text(encoding="utf-8") == content
        assert "v0.8.0" in content
        assert "7aed6b23" in content
        assert "independent project; no endorsement" in content
        assert "Distributed execution: NOT TESTED" in content
        assert "stroke-dasharray" in content
        assert "teacher role" in content
        assert "reference role" in content
        assert "reward role" in content
        assert "export-verl" not in content

    page = (root / "docs" / "verl-bridge.md").read_text(encoding="utf-8")
    assert "<picture" in page
    assert "verl-bridge-architecture-mobile.svg" in page
    assert 'media="(max-width: 600px)"' in page
