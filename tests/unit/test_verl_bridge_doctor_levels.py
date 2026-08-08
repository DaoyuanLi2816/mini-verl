"""Truthful tokenizer verification levels and scoped privacy reporting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tests.conftest import requires_transformers

SECRET = "AKIAIOSFODNN7EXAMPLE"


# ------------------------------------------------------------------ fixtures


def _write_tokenizer_metadata(model: Path) -> None:
    model.mkdir(parents=True, exist_ok=True)
    (model / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "GPT2Tokenizer"}), encoding="utf-8"
    )


def _build_loadable_tokenizer(model: Path, *, extra_special: bool = False) -> None:
    """A complete, tiny, fully local fast-tokenizer snapshot."""
    from tokenizers import Tokenizer, models, pre_tokenizers

    model.mkdir(parents=True, exist_ok=True)
    vocab = {"<unk>": 0, "<pad>": 1, "a": 2, "b": 3, "ab": 4, "hello": 5}
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(model / "tokenizer.json"))
    config: dict[str, Any] = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
    }
    specials: dict[str, Any] = {"unk_token": "<unk>", "pad_token": "<pad>"}
    if extra_special:
        config["eos_token"] = "<pad>"
        specials["eos_token"] = "<pad>"
    (model / "tokenizer_config.json").write_text(json.dumps(config), encoding="utf-8")
    (model / "special_tokens_map.json").write_text(json.dumps(specials), encoding="utf-8")


def _write_manifest(root: Path, identity: dict[str, Any] | None) -> None:
    provenance = root / "provenance"
    provenance.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"schema_version": 1, "run_id": "fixture"}
    if identity is not None:
        payload["tokenizer_identity"] = identity
    (provenance / "miniverl-manifest.json").write_text(json.dumps(payload), encoding="utf-8")


def _identity_of(model: Path) -> dict[str, Any]:
    from transformers import AutoTokenizer

    from miniverl.models.tokenizers import tokenizer_structural_digest

    tokenizer = AutoTokenizer.from_pretrained(
        str(model), local_files_only=True, trust_remote_code=False
    )
    return {
        "structural_digest_v2": tokenizer_structural_digest(tokenizer),
        "vocab_size": len(tokenizer),
        "special_tokens_map": dict(tokenizer.special_tokens_map),
    }


def _check(root: Path, **kwargs: Any) -> dict[str, Any]:
    from miniverl.bridge.doctor import _check_tokenizer

    kwargs.setdefault("require_load", False)
    return _check_tokenizer(root, **kwargs)


# ----------------------------------------------------------- tokenizer levels


def test_absent_tokenizer_reports_not_present(tmp_path: Path) -> None:
    (tmp_path / "model").mkdir(parents=True)
    check = _check(tmp_path)
    assert check["verification_level"] == "not_present"
    assert check["status"] == "fail"
    assert check["structural_identity"] is None


def test_metadata_only_adapter_directory_is_not_called_verified(tmp_path: Path) -> None:
    _write_tokenizer_metadata(tmp_path / "model")
    check = _check(tmp_path)
    assert check["verification_level"] == "metadata_only"
    assert check["status"] == "ok"
    assert check["structural_identity"] is None
    assert any("vocabulary" in item for item in check["missing_components"])
    assert "special_tokens_map.json" in check["missing_components"]
    assert check["load_attempt"] == "not_attempted: tokenizer vocabulary is absent"
    assert "not proof" in check["scope"]


def test_missing_vocabulary_blocks_the_strict_option(tmp_path: Path) -> None:
    _write_tokenizer_metadata(tmp_path / "model")
    check = _check(tmp_path, require_load=True)
    assert check["verification_level"] == "metadata_only"
    assert check["strict_load_satisfied"] is False
    assert check["status"] == "fail"


@requires_transformers
def test_complete_local_snapshot_loads_without_a_reference(tmp_path: Path) -> None:
    _build_loadable_tokenizer(tmp_path / "model")
    check = _check(tmp_path, require_load=True)
    assert check["verification_level"] == "loadable_local_snapshot"
    assert check["load_attempt"] == "passed"
    assert check["status"] == "ok"
    assert check["strict_load_satisfied"] is True
    assert check["missing_components"] == []
    assert check["structural_identity"]["vocab_size"] == 6
    assert check["reference_identity_source"] == "none recorded in the bundle manifest"


@requires_transformers
def test_matching_manifest_identity_reaches_structural_verification(tmp_path: Path) -> None:
    model = tmp_path / "model"
    _build_loadable_tokenizer(model)
    _write_manifest(tmp_path, _identity_of(model))
    check = _check(tmp_path, require_load=True)
    assert check["verification_level"] == "structural_identity_verified"
    assert check["mismatches"] == []
    assert check["status"] == "ok"
    assert check["reference_identity_source"] == "provenance/miniverl-manifest.json"


@requires_transformers
def test_mismatched_tokenizer_fails_closed(tmp_path: Path) -> None:
    model = tmp_path / "model"
    _build_loadable_tokenizer(model)
    identity = _identity_of(model)
    identity["structural_digest_v2"] = "0" * 64
    _write_manifest(tmp_path, identity)
    check = _check(tmp_path)
    assert check["status"] == "fail"
    assert "structural_digest_v2" in check["mismatches"]
    assert check["verification_level"] == "loadable_local_snapshot"


@requires_transformers
def test_changed_special_tokens_fail_closed(tmp_path: Path) -> None:
    reference_root = tmp_path / "reference"
    _build_loadable_tokenizer(reference_root / "model")
    reference = _identity_of(reference_root / "model")

    model = tmp_path / "model"
    _build_loadable_tokenizer(model, extra_special=True)
    _write_manifest(tmp_path, reference)
    check = _check(tmp_path)
    assert check["status"] == "fail"
    assert "special_tokens_map" in check["mismatches"]
    assert "structural_digest_v2" in check["mismatches"]


@requires_transformers
def test_mismatched_vocabulary_size_fails_closed(tmp_path: Path) -> None:
    model = tmp_path / "model"
    _build_loadable_tokenizer(model)
    identity = _identity_of(model)
    identity["vocab_size"] = 999
    _write_manifest(tmp_path, identity)
    check = _check(tmp_path)
    assert check["status"] == "fail"
    assert check["mismatches"] == ["vocab_size"]


@requires_transformers
def test_manifest_without_structural_identity_stays_loadable_only(tmp_path: Path) -> None:
    model = tmp_path / "model"
    _build_loadable_tokenizer(model)
    _write_manifest(tmp_path, {})
    check = _check(tmp_path)
    assert check["verification_level"] == "loadable_local_snapshot"
    assert check["mismatches"] == []
    assert check["status"] == "ok"


@requires_transformers
def test_tokenizer_verification_never_contacts_the_network(
    tmp_path: Path, deny_network: list[str]
) -> None:
    model = tmp_path / "model"
    _build_loadable_tokenizer(model)
    _write_manifest(tmp_path, _identity_of(model))
    check = _check(tmp_path, require_load=True)
    assert check["verification_level"] == "structural_identity_verified"
    assert deny_network == []
    assert "local_files_only=True" in check["network_access"]


def test_corrupt_local_snapshot_reports_a_failed_load(tmp_path: Path) -> None:
    pytest.importorskip("transformers")
    model = tmp_path / "model"
    model.mkdir(parents=True)
    (model / "tokenizer.json").write_text("{ not json", encoding="utf-8")
    (model / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "PreTrainedTokenizerFast"}), encoding="utf-8"
    )
    check = _check(tmp_path)
    assert check["status"] == "fail"
    assert check["load_attempt"].startswith("failed:")
    assert check["verification_level"] == "metadata_only"


# ------------------------------------------------------------ privacy scopes


def _bundle(tmp_path: Path, rows: list[dict[str, Any]] | None = None) -> Path:
    from miniverl.bridge.contract import VERL_TAG
    from miniverl.bridge.export import export_verl_bundle
    from scripts.prepare_verl_bridge_smoke import prepare_smoke_run

    source = prepare_smoke_run(tmp_path / "source")
    if rows is not None:
        table = pa.Table.from_pylist(rows)
        pq.write_table(table, source / "data" / "train.parquet")
        pq.write_table(table, source / "data" / "val.parquet")
    bundle = tmp_path / "bundle"
    export_verl_bundle(source, target_verl=VERL_TAG, out=bundle)
    return bundle


def _rows(content: str, count: int = 1) -> list[dict[str, Any]]:
    return [
        {
            "data_source": "miniverl-bridge-smoke",
            "prompt": [{"role": "user", "content": content}],
            "ability": "formatting",
            "reward_model": {"style": "rule", "ground_truth": "4"},
            "extra_info": {"split": "smoke", "synthetic": True},
        }
        for _ in range(count)
    ]


def test_default_doctor_never_claims_dataset_or_weight_inspection(tmp_path: Path) -> None:
    from miniverl.bridge.doctor import inspect_bridge_bundle

    payload = inspect_bridge_bundle(_bundle(tmp_path))
    assert payload["verdict"] == "ok"
    assert payload["portable_metadata_privacy"] == "heuristic_passed_full"
    assert payload["dataset_content_privacy"] == "not_inspected"
    assert payload["model_weight_privacy"] == "not_inspected"
    privacy = payload["privacy"]
    assert privacy["status"] == "ok"
    assert privacy["dataset_scan"]["status"] == "not_inspected"
    assert privacy["dataset_scan"]["findings"] == []
    assert "not_inspected never means passed" in privacy["scope_note"]
    # The default bundle is metadata-only, and the report must say so.
    assert payload["tokenizer_verification_level"] == "metadata_only"


def test_clean_dataset_scan_reports_passed_not_assumed(tmp_path: Path) -> None:
    from miniverl.bridge.doctor import inspect_bridge_bundle

    payload = inspect_bridge_bundle(_bundle(tmp_path), scan_dataset_text=True)
    scan = payload["privacy"]["dataset_scan"]
    assert payload["dataset_content_privacy"] == "passed"
    assert scan["findings"] == []
    assert scan["scan_scope"] == "full"
    assert scan["rows_scanned"] == scan["rows_total"] == 2
    assert "not de-identification proof" in scan["method"]
    assert payload["model_weight_privacy"] == "not_inspected"


def test_dataset_scan_detects_a_credential_without_disclosing_it(tmp_path: Path) -> None:
    from miniverl.bridge.doctor import inspect_bridge_bundle

    bundle = _bundle(tmp_path, _rows(f"my key is {SECRET} keep it safe"))
    payload = inspect_bridge_bundle(bundle, scan_dataset_text=True)

    assert payload["dataset_content_privacy"] == "failed"
    assert payload["verdict"] == "fail"
    findings = payload["privacy"]["dataset_scan"]["findings"]
    assert {item["category"] for item in findings} == {"aws_access_key_id"}
    assert findings[0]["column"] == "prompt"
    assert findings[0]["row"] == 0
    assert set(findings[0]) == {"category", "split", "column", "row"}
    # The matched value must never leave the scanner, in any field.
    assert SECRET not in json.dumps(payload)


def test_dataset_scan_detects_a_user_sentinel(tmp_path: Path) -> None:
    from miniverl.bridge.doctor import inspect_bridge_bundle

    bundle = _bundle(tmp_path, _rows("nothing unusual but codeword banana here"))
    payload = inspect_bridge_bundle(bundle, scan_dataset_text=True, sentinels=("codeword banana",))
    findings = payload["privacy"]["dataset_scan"]["findings"]
    assert {item["category"] for item in findings} == {"user_sentinel"}
    assert "codeword banana" not in json.dumps(payload)
    assert "user_sentinel" in payload["privacy"]["dataset_scan"]["detectors"]


def test_dataset_scan_is_bounded_and_records_that_it_sampled(tmp_path: Path) -> None:
    from miniverl.bridge.doctor import inspect_bridge_bundle

    bundle = _bundle(tmp_path, _rows("harmless prompt", count=5))
    payload = inspect_bridge_bundle(bundle, scan_dataset_text=True, dataset_scan_max_rows=2)
    scan = payload["privacy"]["dataset_scan"]
    assert scan["scan_scope"] == "sampled"
    assert scan["rows_scanned"] == 2
    assert scan["rows_total"] == 10
    assert scan["max_rows"] == 2


def test_dataset_scan_never_reads_safetensors_as_text(tmp_path: Path) -> None:
    from miniverl.bridge.doctor import inspect_bridge_bundle

    bundle = _bundle(tmp_path)
    payload = inspect_bridge_bundle(bundle, scan_dataset_text=True)
    scan = payload["privacy"]["dataset_scan"]
    assert scan["status"] == "passed"
    assert all(item["split"] in {"train", "val"} for item in scan["findings"])


def test_privacy_report_serializes_as_canonical_machine_json(tmp_path: Path) -> None:
    from miniverl.bridge.doctor import inspect_bridge_bundle
    from miniverl.utils.runs import canonical_json

    payload = inspect_bridge_bundle(_bundle(tmp_path), scan_dataset_text=True)
    restored = json.loads(canonical_json(payload))
    assert restored["privacy"]["portable_metadata_privacy"] == "heuristic_passed_full"
    assert restored["privacy"]["model_weight_privacy"] == "not_inspected"
    assert restored["tokenizer_identity"]["verification_level"] == "metadata_only"


def test_metadata_privacy_failure_is_scoped_to_metadata(tmp_path: Path) -> None:
    from miniverl.bridge.doctor import inspect_bridge_bundle

    bundle = _bundle(tmp_path)
    (bundle / "recipe" / "leak.txt").write_text("C:\\Users\\someone\\secret", encoding="utf-8")
    payload = inspect_bridge_bundle(bundle)
    assert payload["portable_metadata_privacy"] == "heuristic_failed"
    assert payload["dataset_content_privacy"] == "not_inspected"
    assert payload["privacy"]["problems"] == ["recipe/leak.txt"]
