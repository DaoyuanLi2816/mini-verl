"""A bundle's own claims are never promoted to locally verified facts.

``provenance/SHA256SUMS`` lives inside the bundle it describes. An attacker who
edits ``compatibility-report.json`` can regenerate the checksum file just as
easily, so agreement between them proves internal consistency and nothing more.

Up to the v0.6.3 release candidate the doctor read
``upstream_config_parse_passed``, ``model_data_load_smoke_passed``,
``distributed_execution_tested`` and ``algorithm_semantic_parity`` straight out
of that report and exposed them as top-level results, which read as verified
outcomes of the diagnosis that just ran.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from tests.unit.test_verl_bridge_hostile_bundles import _run

LIES: dict[str, Any] = {
    "upstream_config_parse_passed": True,
    "model_data_load_smoke_passed": True,
    "distributed_execution_tested": True,
    "algorithm_semantic_parity": True,
    "distributed_execution_status": "verified on 512 GPUs",
    "launchable": True,
    "unsupported_semantics": [],
}


def _reseal(bundle: Path) -> None:
    """Rewrite SHA256SUMS so the tampered bundle is internally consistent."""
    checksum = bundle / "provenance" / "SHA256SUMS"
    lines = []
    for path in sorted(bundle.rglob("*")):
        if not path.is_file() or path == checksum:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(bundle).as_posix()}")
    checksum.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _lying_bundle(tmp_path: Path) -> Path:
    from miniverl.bridge.contract import VERL_TAG
    from miniverl.bridge.export import export_verl_bundle

    out = tmp_path / "export"
    export_verl_bundle(_run(tmp_path), target_verl=VERL_TAG, out=out)
    report = out / "provenance" / "compatibility-report.json"
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload.update(LIES)
    report.write_text(json.dumps(payload), encoding="utf-8")
    _reseal(out)
    return out


def test_self_consistent_hashes_do_not_make_claims_trustworthy(tmp_path: Path) -> None:
    from miniverl.bridge.doctor import inspect_bridge_bundle

    diagnosis = inspect_bridge_bundle(_lying_bundle(tmp_path))

    # The bundle really is internally consistent -- that is the whole point.
    assert diagnosis["artifact_hashes"]["status"] == "ok"
    assert diagnosis["provenance_trust"]["level"] == "unsigned_self_consistent"
    assert diagnosis["provenance_trust"]["signature_verification"] == "not_available"

    declared = diagnosis["bundle_declared_claims"]
    assert declared["upstream_config_parse_passed"] is True
    assert declared["distributed_execution_tested"] is True
    assert declared["algorithm_semantic_parity"] is True
    assert declared["trust"] == "unsigned_self_consistent"

    # None of it is promoted.
    assert diagnosis["upstream_config_parse_passed"] is False
    assert diagnosis["model_data_load_smoke_passed"] is False
    assert diagnosis["distributed_execution_tested"] is False
    assert diagnosis["algorithm_semantic_parity"] is False
    assert diagnosis["launchable"] is False
    assert diagnosis["distributed_execution_status"] == "not tested"


def test_locally_recomputed_checks_are_listed_separately(tmp_path: Path) -> None:
    from miniverl.bridge.doctor import inspect_bridge_bundle

    diagnosis = inspect_bridge_bundle(_lying_bundle(tmp_path))
    local = diagnosis["locally_recomputed_checks"]

    # Recomputed in this process from the bytes on disk.
    for name in (
        "checksum_consistency",
        "pinned_requirement_file",
        "config_structure",
        "adapter_safetensors_structure",
        "tokenizer_identity",
        "parquet_schema",
        "reward_interface_static",
        "portable_metadata_privacy",
    ):
        assert name in local, name
        assert local[name] in {"passed", "failed"}, name

    # Never recomputed by a doctor run.
    assert local["upstream_config_parse"] == "not_run"
    assert local["distributed_execution"] == "not_run"
    assert local["algorithm_semantic_parity"] == "not_run"


def test_upstream_smoke_claim_is_not_recomputed_without_require_verl(tmp_path: Path) -> None:
    from miniverl.bridge.doctor import inspect_bridge_bundle

    diagnosis = inspect_bridge_bundle(_lying_bundle(tmp_path))

    assert diagnosis["locally_recomputed_checks"]["upstream_config_parse"] == "not_run"
    assert diagnosis["installed_verl"]["status"] in {"not installed", "unverified", "ok"}


def test_declared_claims_are_absent_rather_than_false_when_no_report_exists(
    tmp_path: Path,
) -> None:
    from miniverl.bridge.contract import VERL_TAG
    from miniverl.bridge.doctor import inspect_bridge_bundle
    from miniverl.bridge.export import export_verl_bundle

    out = tmp_path / "export"
    export_verl_bundle(_run(tmp_path), target_verl=VERL_TAG, out=out)
    (out / "provenance" / "compatibility-report.json").unlink()
    _reseal(out)

    diagnosis = inspect_bridge_bundle(out)

    assert diagnosis["bundle_declared_claims"]["source"] == "absent"
    assert diagnosis["bundle_declared_claims"]["trust"] == "not_verified"
    assert diagnosis["upstream_config_parse_passed"] is False


def test_report_never_calls_the_bundle_trusted(tmp_path: Path) -> None:
    from miniverl.bridge.doctor import inspect_bridge_bundle

    diagnosis = inspect_bridge_bundle(_lying_bundle(tmp_path))
    note = diagnosis["provenance_trust"]["note"].lower()

    assert "internal consistency" in note
    assert "trusted" not in diagnosis["provenance_trust"]["level"]
