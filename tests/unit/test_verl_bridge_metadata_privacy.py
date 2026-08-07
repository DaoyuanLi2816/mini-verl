"""Portable metadata privacy covers credentials, not only absolute paths.

Up to the v0.6.3 release candidate the default metadata scan ran exactly one
detector -- an absolute-path regex -- so a manifest carrying an API key, a
bearer token and a database URL with inline credentials passed as
``portable_metadata_privacy: passed``.

Every value below is a fabricated test fixture, not a real credential.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

FAKE_API_KEY = "FAKE_SUPER_SECRET_1234567890"
FAKE_BEARER = "Bearer abcdefghijklmnopqrstuvwxyz012345"
FAKE_DB_URL = "postgresql://user:password@example.com/db"
FAKE_AWS = "AKIAIOSFODNN7EXAMPLE"


def _scan(root: Path, **kwargs: Any) -> dict[str, Any]:
    from miniverl.bridge.doctor import _check_privacy

    return _check_privacy(root, **kwargs)


def _bundle(tmp_path: Path, payload: dict[str, Any]) -> Path:
    provenance = tmp_path / "provenance"
    provenance.mkdir(parents=True, exist_ok=True)
    (provenance / "miniverl-manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    return tmp_path


# --------------------------------------------------------------- structured data


def test_credentials_in_json_metadata_are_detected(tmp_path: Path) -> None:
    root = _bundle(
        tmp_path,
        {
            "api_key": FAKE_API_KEY,
            "authorization": FAKE_BEARER,
            "database_url": FAKE_DB_URL,
        },
    )

    privacy = _scan(root)

    assert privacy["portable_metadata_privacy"] == "heuristic_failed"
    assert privacy["status"] == "fail"
    categories = {finding["category"] for finding in privacy["metadata_scan"]["findings"]}
    assert "semantic_secret_key" in categories
    assert "url_userinfo" in categories


def test_findings_report_the_json_path_but_never_the_value(tmp_path: Path) -> None:
    root = _bundle(tmp_path, {"nested": {"credentials": {"api_key": FAKE_API_KEY}}})

    privacy = _scan(root)
    findings = privacy["metadata_scan"]["findings"]

    assert findings
    assert any(finding["path"] == "$.nested.credentials.api_key" for finding in findings)
    assert FAKE_API_KEY not in json.dumps(privacy)


def test_secret_inside_a_list_reports_an_indexed_path(tmp_path: Path) -> None:
    root = _bundle(tmp_path, {"items": [{"token": FAKE_API_KEY}]})

    privacy = _scan(root)

    assert any(
        finding["path"] == "$.items[0].token" for finding in privacy["metadata_scan"]["findings"]
    )


def test_aws_style_key_in_a_value_is_detected(tmp_path: Path) -> None:
    root = _bundle(tmp_path, {"note": f"deploy with {FAKE_AWS} please"})

    privacy = _scan(root)

    categories = {finding["category"] for finding in privacy["metadata_scan"]["findings"]}
    assert "aws_access_key_id" in categories
    assert FAKE_AWS not in json.dumps(privacy)


def test_yaml_metadata_is_scanned_too(tmp_path: Path) -> None:
    recipe = tmp_path / "recipe"
    recipe.mkdir(parents=True)
    (recipe / "verl-overrides.yaml").write_text(f"trainer:\n  api_key: {FAKE_API_KEY}\n", "utf-8")

    privacy = _scan(tmp_path)

    assert privacy["portable_metadata_privacy"] == "heuristic_failed"
    assert any(
        finding["path"] == "$.trainer.api_key" for finding in privacy["metadata_scan"]["findings"]
    )


# ------------------------------------------------------------- unstructured text


def test_markdown_secret_reports_a_line_number_not_the_value(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text(
        f"# Bundle\n\nRun with\n\n    export API_TOKEN={FAKE_API_KEY}\n", encoding="utf-8"
    )

    privacy = _scan(tmp_path)
    findings = privacy["metadata_scan"]["findings"]

    assert findings
    assert all(finding.get("line") != 0 for finding in findings)
    assert FAKE_API_KEY not in json.dumps(privacy)


def test_absolute_local_path_is_still_detected(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("saved to C:\\Users\\someone\\secret\n", encoding="utf-8")

    privacy = _scan(tmp_path)

    categories = {finding["category"] for finding in privacy["metadata_scan"]["findings"]}
    assert "absolute_local_path" in categories


def test_user_sentinels_are_honoured(tmp_path: Path) -> None:
    root = _bundle(tmp_path, {"comment": "contains CANARY-99 somewhere"})

    privacy = _scan(root, sentinels=("CANARY-99",))

    categories = {finding["category"] for finding in privacy["metadata_scan"]["findings"]}
    assert "user_sentinel" in categories
    assert "CANARY-99" not in json.dumps(privacy)


# ------------------------------------------------------------------------ bounds


def test_clean_metadata_passes_with_the_heuristic_name(tmp_path: Path) -> None:
    root = _bundle(tmp_path, {"run_id": "fixture", "model_id": "Qwen/Qwen3-0.6B"})

    privacy = _scan(root)

    assert privacy["portable_metadata_privacy"] == "heuristic_passed_full"
    assert privacy["status"] == "ok"
    assert privacy["metadata_scan"]["findings"] == []
    assert "de-identification" in privacy["metadata_scan"]["method"]


def test_oversized_metadata_file_is_skipped_and_recorded(tmp_path: Path) -> None:
    from miniverl.bridge.doctor import METADATA_MAX_FILE_BYTES

    (tmp_path / "huge.txt").write_text("A" * (METADATA_MAX_FILE_BYTES + 10), encoding="utf-8")

    privacy = _scan(tmp_path)

    assert privacy["metadata_scan"]["files_skipped_too_large"] == 1
    assert privacy["metadata_scan"]["scan_scope"] == "sampled"


def test_findings_are_bounded(tmp_path: Path) -> None:
    from miniverl.bridge.doctor import METADATA_MAX_FINDINGS

    payload = {
        f"entry_{index}": {"api_key": FAKE_API_KEY} for index in range(METADATA_MAX_FINDINGS + 50)
    }
    root = _bundle(tmp_path, payload)

    privacy = _scan(root)

    assert len(privacy["metadata_scan"]["findings"]) <= METADATA_MAX_FINDINGS
    assert privacy["metadata_scan"]["findings_truncated"] is True


def test_binary_payloads_are_never_read_as_text(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "train.parquet").write_bytes(b"PAR1" + FAKE_API_KEY.encode() + b"PAR1")
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "adapter_model.safetensors").write_bytes(b"\x00" * 32)

    privacy = _scan(tmp_path)

    assert privacy["portable_metadata_privacy"] == "heuristic_passed_full"
    assert privacy["dataset_content_privacy"] == "not_inspected"
    assert privacy["model_weight_privacy"] == "not_inspected"


@pytest.mark.parametrize("scope", ["dataset_content_privacy", "model_weight_privacy"])
def test_uninspected_scopes_never_become_passed(tmp_path: Path, scope: str) -> None:
    root = _bundle(tmp_path, {"run_id": "fixture"})

    privacy = _scan(root)

    assert privacy[scope] == "not_inspected"
