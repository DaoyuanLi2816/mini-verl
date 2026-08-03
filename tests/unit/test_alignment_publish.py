"""Frozen Alignment Lab publication and cost-accounting contracts."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import jsonschema
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


def _publisher() -> ModuleType:
    path = ROOT / "scripts" / "publish_alignment_lab_artifacts.py"
    spec = importlib.util.spec_from_file_location("publish_alignment_lab_artifacts", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_alignment_preregistration_and_immutable_calculator_are_frozen() -> None:
    preregistration = ROOT / "benchmarks/preregistration/alignment-lab-v1.yaml"
    assert hashlib.sha256(preregistration.read_bytes()).hexdigest() == (
        "71307dbfe9a5bb20c686307cafce8bd254c07af8b69c1bf1c6ec0dbf53a8cde0"
    )
    payload = yaml.safe_load(preregistration.read_text(encoding="utf-8"))
    assert payload["preregistration_revision"] == 1.4
    assert payload["execution"]["student_seeds"] == [1234, 20260727, 20260801]
    assert payload["final_test"] == {
        "split": "test",
        "tasks": 48,
        "read_count": 1,
        "timing": "only_after_this_preregistration_commit_is_public",
        "temperature": 0.0,
        "threshold_changes_after_read": "forbidden",
        "metric_replacement_after_read": "forbidden",
    }
    calculator = ROOT / "benchmarks/results/gpu-calc-hard-equal-update-v2.json"
    assert hashlib.sha256(calculator.read_bytes()).hexdigest() == (
        "53fc1d4d5b7adee09618d77ad62d4086ba56b78569832d6fc7c3bcd5c2695bbc"
    )


def test_alignment_result_schema_and_paired_task_evidence() -> None:
    result_path = ROOT / "benchmarks/results/alignment-lab-v1.json"
    task_path = ROOT / "benchmarks/results/alignment-lab-v1-task-results.jsonl"
    schema = json.loads(
        (ROOT / "benchmarks/schema/alignment-lab-result.schema.json").read_text(encoding="utf-8")
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == (
        "584752dccb91654109c357b8ebb12681a12a9c1476a9ba539dd35e4d860a22ef"
    )
    assert hashlib.sha256(task_path.read_bytes()).hexdigest() == (
        "8d7fc723436d7377d196fc44046d960e3cb7f0aa81e03d49ef05b627eb84630f"
    )
    jsonschema.validate(result, schema)
    assert result["measurement_status"] == "measured_final"
    assert len(result["arms"]) == 18
    assert {arm["seed"] for arm in result["arms"]} == {1234, 20260727, 20260801}
    assert all(arm["metrics"]["tasks"] == 48 for arm in result["arms"])
    assert all(arm["status"] == "completed" for arm in result["arms"])
    assert all(
        arm["starting_checkpoint_sha256"] == result["starting_checkpoint_sha256"]
        for arm in result["arms"]
    )

    rows = [json.loads(line) for line in task_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 18 * 48
    by_seed_method: dict[tuple[int, str], list[str]] = {}
    for row in rows:
        by_seed_method.setdefault((row["seed"], row["method"]), []).append(row["task_id"])
    for seed in (1234, 20260727, 20260801):
        task_sets = {
            tuple(sorted(task_ids))
            for (row_seed, _), task_ids in by_seed_method.items()
            if row_seed == seed
        }
        assert len(task_sets) == 1
    assert result["task_results_sha256"] == hashlib.sha256(task_path.read_bytes()).hexdigest()
    diagnostic_path = ROOT / "benchmarks/results/alignment-lab-v1-state-supervision.json"
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    diagnostic_schema = json.loads(
        (ROOT / "benchmarks/schema/alignment-state-supervision.schema.json").read_text(
            encoding="utf-8"
        )
    )
    jsonschema.validate(diagnostic, diagnostic_schema)
    assert diagnostic["measurement_status"] == "measured_signal_diagnostic_not_training_outcome"
    assert len(diagnostic["cells"]) == 4
    matched = diagnostic["matched_comparisons"]["fresh_hard_vs_fresh_soft"]
    assert all(
        matched[key]
        for key in (
            "same_state_digest",
            "same_teacher_digest",
            "same_budget_digest",
            "same_starting_checkpoint_digest",
            "same_seeds",
        )
    )
    assert diagnostic["claims"]["state_or_soft_target_quality_advantage"] == "not_claimed"
    assert (
        result["state_supervision_diagnostic"]["sha256"]
        == hashlib.sha256(diagnostic_path.read_bytes()).hexdigest()
    )
    assert result["pilot"]["recommendation"] == "insufficient_evidence"
    assert result["pilot"]["evidence"]["teacher_policy_competence"] is None
    assert "headroom" in result["pilot"]["reasons"][0]
    pilot_example = ROOT / "examples/alignment-lab/pilot.json"
    assert json.loads(pilot_example.read_text(encoding="utf-8")) == result["pilot"]


def test_alignment_figures_are_exactly_generated_and_privacy_safe() -> None:
    publisher = _publisher()
    result_path = ROOT / "benchmarks/results/alignment-lab-v1.json"
    payload = publisher._load_result(result_path)
    source_digest = hashlib.sha256(result_path.read_bytes()).hexdigest()
    rendered = publisher.render_figures(payload, source_digest)
    assert set(rendered) == {
        "delta-from-sft.svg",
        "outcome-cost-matrix.svg",
        "metric-coverage-matrix.svg",
    }
    publisher.assert_chart_suitability(rendered)
    combined_svg = "\n".join(rendered.values())
    assert "concentric" not in combined_svg.lower()
    assert "jitter" not in combined_svg.lower()
    assert source_digest not in combined_svg
    assert "—  not applicable" in rendered["outcome-cost-matrix.svg"]
    assert 'data-applicable="false"' in rendered["outcome-cost-matrix.svg"]
    assert 'data-encoding="seed-point"' in combined_svg
    assert (
        "The two sandbox safety checks tied at zero while utility still regressed."
        in (rendered["metric-coverage-matrix.svg"])
    )
    for name, content in rendered.items():
        target = ROOT / "docs/alignment-lab" / name
        assert target.read_text(encoding="utf-8") == content
    cards = publisher.render_cards(payload, source_digest)
    assert len(cards) == 36
    for name, content in cards.items():
        target = ROOT / "benchmarks/alignment-cards/alignment-lab-v1" / name
        assert target.read_text(encoding="utf-8") == content
    report = publisher.render_report(payload, source_digest)
    assert (ROOT / "docs/alignment-lab/alignment-lab-v1.md").read_text(encoding="utf-8") == report
    combined = (
        result_path.read_text(encoding="utf-8")
        + "\n"
        + (ROOT / "benchmarks/results/alignment-lab-v1-task-results.jsonl").read_text(
            encoding="utf-8"
        )
    )
    assert "C:\\Users\\" not in combined
    assert "/home/" not in combined

    pdf_path = ROOT / "paper" / "alignment-lab-v1" / "alignment-lab-v1.pdf"
    assert hashlib.sha256(pdf_path.read_bytes()).hexdigest() == (
        "db4aeb2507839ce200cb5c6d93855fd687b9bb8b978fe76b089c5f1993af78a5"
    )


def test_alignment_svg_titles_are_portable_across_fallback_fonts() -> None:
    publisher = _publisher()
    with pytest.raises(ValueError, match="portable across deterministic fallback fonts"):
        publisher._svg_shell("x" * 49, "description", "subtitle", [])


def test_dpo_external_training_cost_is_included() -> None:
    result = json.loads(
        (ROOT / "benchmarks/results/alignment-lab-v1.json").read_text(encoding="utf-8")
    )
    dpo = [arm for arm in result["arms"] if arm["method"] == "dpo"]
    assert len(dpo) == 3
    assert all(arm["cost"]["optimizer_updates"] == 4 for arm in dpo)
    assert all(arm["cost"]["gpu_seconds"] > 8.0 for arm in dpo)
    assert all(arm["cost"]["peak_vram_bytes"] > 1_700_000_000 for arm in dpo)
    assert all(arm["provenance"]["trl_version"] == "1.8.0" for arm in dpo)
    for arm in dpo:
        identity = arm["provenance"]["dpo_identity"]
        assert identity["reference"]["adapter_weights_sha256"]
        assert identity["dataset"]["sha256"]
        assert identity["checkpoint"]["sha256"] == result["starting_checkpoint_sha256"]
        assert identity["adapter"]["weights_sha256"] == arm["provenance"]["adapter_weights_sha256"]

        card = json.loads(
            (
                ROOT
                / "benchmarks/alignment-cards/alignment-lab-v1"
                / f"dpo-seed-{arm['seed']}.json"
            ).read_text(encoding="utf-8")
        )
        assert card["reference"]["adapter"]["adapter_weights_sha256"]
        assert card["dpo_training"] == identity


def test_recovered_seed_1234_baseline_has_two_disjoint_preserved_segments() -> None:
    result = json.loads(
        (ROOT / "benchmarks/results/alignment-lab-v1.json").read_text(encoding="utf-8")
    )
    arm = next(
        item
        for item in result["arms"]
        if item["method"] == "sft_checkpoint" and item["seed"] == 1234
    )
    assert arm["recovery"]["kind"] == "two_disjoint_segments"
    assert [segment["task_offset"] for segment in arm["recovery"]["segments"]] == [0, 24]
    assert [segment["tasks"] for segment in arm["recovery"]["segments"]] == [24, 24]
    assert arm["metrics"]["tasks"] == 48
