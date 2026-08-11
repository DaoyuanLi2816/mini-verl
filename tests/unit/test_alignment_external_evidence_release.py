"""v0.7 external-alignment early-stop evidence is explicit and self-consistent."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import jsonschema
import pytest
from pydantic import ValidationError

from miniverl.alignment_external.result import AlignmentExternalResult

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "benchmarks/results/alignment-external-v1.json"


def _selection_script() -> Any:
    path = ROOT / "scripts/select_external_alignment_start.py"
    spec = importlib.util.spec_from_file_location("select_external_alignment_start", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _publisher() -> Any:
    path = ROOT / "scripts/publish_alignment_external_artifacts.py"
    spec = importlib.util.spec_from_file_location("publish_alignment_external_artifacts", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_lineage_metadata_is_required_before_heavy_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _selection_script()
    monkeypatch.setattr(sys, "argv", ["select", "--candidates", "c", "--out", "o"])

    with pytest.raises(SystemExit):
        module._parse_args()

    assert "torch" not in module.__dict__


def test_lineage_metadata_is_typed_and_not_hard_coded(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _selection_script()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "select",
            "--candidates",
            "c",
            "--out",
            "o",
            "--lineage-id",
            "fallback",
            "--lineage-description",
            "amendment 2 anchor continued on HH-RLHF",
            "--lineage-anchor",
            "repo@revision",
        ],
    )

    args = module._parse_args()

    assert args.lineage_id == "fallback"
    assert args.lineage_description.startswith("amendment 2")
    assert args.lineage_anchor == "repo@revision"
    assert "primary: Qwen3-0.6B" not in Path(module.__file__).read_text(encoding="utf-8")


def test_committed_early_stop_result_validates() -> None:
    result = AlignmentExternalResult.model_validate_json(RESULT.read_text(encoding="utf-8"))

    assert result.study_status == "terminated_at_checkpoint_selection"
    assert result.selected_checkpoint is None
    assert len(result.checkpoint_selection.lineages) == 2
    assert sum(len(lineage.candidates) for lineage in result.checkpoint_selection.lineages) == 8

    schema = json.loads(
        (ROOT / "benchmarks/schema/alignment-external-result.schema.json").read_text(
            encoding="utf-8"
        )
    )
    jsonschema.validate(result.model_dump(mode="json"), schema)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda row: row.update(selected_checkpoint="update-004"),
        lambda row: row["final_test"].update(tasks_scored=1),
        lambda row: row["teacher_qualification"].update(status="completed"),
        lambda row: row["continuation_methods"].update(status="completed"),
        lambda row: row["evaluators"]["granite_guardian"].update(headline_eligible=True),
    ],
)
def test_impossible_early_stop_states_are_rejected(mutation: Any) -> None:
    payload = json.loads(RESULT.read_text(encoding="utf-8"))
    mutation(payload)

    with pytest.raises(ValidationError):
        AlignmentExternalResult.model_validate(payload)


def test_metadata_correction_preserves_every_quantitative_field() -> None:
    original = json.loads(
        (
            ROOT
            / "benchmarks/evidence/alignment-external-v1/fallback-start-selection.original.json"
        ).read_text(encoding="utf-8")
    )
    corrected = json.loads(
        (
            ROOT
            / "benchmarks/evidence/alignment-external-v1/fallback-start-selection.corrected.json"
        ).read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (
            ROOT / "benchmarks/evidence/alignment-external-v1/fallback-correction-manifest.json"
        ).read_text(encoding="utf-8")
    )

    assert original["candidate_results"] == corrected["candidate_results"]
    assert original["decision"] == corrected["decision"]
    assert manifest["correction"]["quantitative_values_changed"] is False
    assert manifest["correction"]["selection_decision_changed"] is False


def test_selection_suites_are_identical_but_not_independent() -> None:
    disclosure = json.loads(
        (
            ROOT / "benchmarks/evidence/alignment-external-v1/selection-suite-disclosure.json"
        ).read_text(encoding="utf-8")
    )

    assert disclosure["separately_generated"] is True
    assert disclosure["task_ids_identical"] is True
    assert disclosure["independent_task_set"] is False
    assert (
        disclosure["primary_selection_suite"]["sha256"]
        == disclosure["fallback_selection_suite"]["sha256"]
    )
    assert disclosure["final_test_disjoint"] is True


def test_portable_jsonnav_evidence_has_all_rows_without_private_paths() -> None:
    path = ROOT / "benchmarks/evidence/alignment-external-v1/jsonnav-selection-records.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    assert len(rows) == 8 * 64
    assert {row["lineage_id"] for row in rows} == {"primary", "fallback"}
    assert all(row["solved"] is False for row in rows)
    primary = [row for row in rows if row["lineage_id"] == "primary"]
    fallback = [row for row in rows if row["lineage_id"] == "fallback"]
    assert all(row["tool_call_count"] == 0 for row in primary)
    assert all(row["tool_call_count"] >= 2 for row in fallback)
    assert "C:\\Users\\" not in path.read_text(encoding="utf-8")

    schema = json.loads(
        (ROOT / "benchmarks/schema/alignment-external-selection-task.schema.json").read_text(
            encoding="utf-8"
        )
    )
    for row in rows:
        jsonschema.validate(row, schema)


def test_superseded_log_projection_preserves_source_digest_without_private_paths() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    artifact = result["superseded_proxy_artifact"]
    projected = ROOT / artifact["path"]
    text = projected.read_text(encoding="utf-8")

    assert "C:\\Users\\" not in text
    assert "<repository>\\scripts\\select_external_alignment_start.py" in text
    assert artifact["projection"] == "absolute_paths_replaced"
    assert artifact["source_sha256"] == (
        "9efd0bbc3f74c93e6cef8ced00de65796230eaada2838c94026e168b871a26af"
    )
    assert hashlib.sha256(projected.read_bytes()).hexdigest() == artifact["sha256"]


def test_generated_artifacts_and_figures_are_byte_identical() -> None:
    publisher = _publisher()
    tracked = [
        RESULT,
        ROOT / "benchmarks/schema/alignment-external-result.schema.json",
        ROOT / "benchmarks/schema/alignment-external-selection-task.schema.json",
        ROOT / "benchmarks/evidence/alignment-external-v1/fallback-start-selection.original.json",
        ROOT / "benchmarks/evidence/alignment-external-v1/primary-start-selection.original.json",
        ROOT / "benchmarks/evidence/alignment-external-v1/primary-jsonnav-records.original.json",
        ROOT / "benchmarks/evidence/alignment-external-v1/fallback-jsonnav-records.original.json",
        ROOT / "benchmarks/evidence/alignment-external-v1/primary-selection-suite.original.json",
        ROOT / "benchmarks/evidence/alignment-external-v1/fallback-selection-suite.original.json",
        ROOT / "benchmarks/evidence/alignment-external-v1/primary-final-suite.original.json",
        ROOT / "benchmarks/evidence/alignment-external-v1/fallback-final-suite.original.json",
        ROOT / "benchmarks/evidence/alignment-external-v1/superseded-pre-amendment-run.log",
        ROOT / "benchmarks/evidence/alignment-external-v1/fallback-start-selection.corrected.json",
        ROOT / "benchmarks/evidence/alignment-external-v1/fallback-correction-manifest.json",
        ROOT / "benchmarks/evidence/alignment-external-v1/selection-suite-disclosure.json",
        ROOT / "benchmarks/evidence/alignment-external-v1/jsonnav-selection-records.jsonl",
        ROOT / "benchmarks/evidence/alignment-external-v1/jsonnav-selection-records.manifest.json",
        ROOT / "docs/alignment-external/checkpoint-gate-matrix.svg",
        ROOT / "docs/alignment-external/checkpoint-gate-matrix-mobile.svg",
        ROOT / "docs/alignment-external/study-early-stop.svg",
        ROOT / "docs/alignment-external/study-early-stop-mobile.svg",
    ]
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in tracked}

    publisher.publish()

    after = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in tracked}
    assert after == before
    for path in tracked[-4:]:
        content = path.read_text(encoding="utf-8")
        assert "<title>" in content and "<desc>" in content
        assert "jitter" not in content.lower()
        assert "concentric" not in content.lower()
