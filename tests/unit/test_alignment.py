from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from miniverl.alignment import (
    AlignmentMethod,
    AlignmentMetrics,
    ArtifactIdentity,
    PilotEvidence,
    PilotRecommendation,
    StateSource,
    StateSupervisionRecord,
    Supervision,
    alignment_metrics,
    build_alignment_stage_plan,
    build_tool_policy_preferences,
    load_benchmark_registry,
    preference_dataset_digest,
    recommend_alignment_method,
    render_alignment_card,
    validate_state_supervision_matrix,
)
from miniverl.config.models import AlignmentConfig, GateConfig, GateSignal, TeacherMode
from miniverl.schemas.trajectory import (
    Span,
    SpanType,
    TerminationReason,
    Trajectory,
    Turn,
    VerificationRecord,
)
from miniverl.trajectory.masks import build_masks


def _trajectory(
    task_id: str,
    *,
    expected: str,
    predicted: str,
    solved: bool,
    category: str,
    benign: bool,
    harmful: bool = False,
    tool_utility: bool = False,
    tag: str = "final",
) -> Trajectory:
    spans = [
        Span(span_type=SpanType.SYSTEM, start=0, end=1, turn_id=0, text="system"),
        Span(span_type=SpanType.USER, start=1, end=2, turn_id=0, text="user"),
        Span(span_type=SpanType.ASSISTANT_FINAL, start=2, end=3, turn_id=0, text=predicted),
    ]
    model, critical = build_masks(spans, 3)
    return Trajectory(
        trajectory_id=f"{task_id}:{tag}:v0",
        task_id=task_id,
        environment="tool_policy",
        token_ids=[1, 2, 3],
        attention_mask=[1, 1, 1],
        model_generated_mask=model,
        critical_mask=critical,
        spans=spans,
        turns=[Turn(turn_id=0, is_final=True)],
        tokenizer_fingerprint="fp",
        model_id="model",
        verification=VerificationRecord(
            solved=solved,
            reward=float(solved),
            expected=expected,
            predicted=predicted,
            failure_category="solved" if solved else "wrong_answer",
        ),
        termination_reason=TerminationReason.FINAL_ANSWER,
        generated_token_count=1,
        metadata={
            "policy_category": category,
            "expected_decision": expected,
            "benign": benign,
            "harmful": harmful,
            "tool_utility": tool_utility,
            "instruction_retention": benign,
        },
    )


def test_alignment_config_requires_a_frozen_gate_for_verifier_gated_opd() -> None:
    with pytest.raises(ValidationError, match="gate"):
        AlignmentConfig(
            method=AlignmentMethod.VERIFIER_GATED_OPD,
            teacher_mode=TeacherMode.POLICY_CONDITIONED,
            starting_sft_checkpoint="sft/final",
            starting_sft_checkpoint_sha256="b" * 64,
            policy=ArtifactIdentity(id="mini-policy", revision="v1", sha256="a" * 64),
        )


def test_alignment_stage_plan_names_every_auditable_stage_without_local_paths() -> None:
    alignment = AlignmentConfig(
        method=AlignmentMethod.STANDARD_OPD,
        teacher_mode=TeacherMode.POLICY_CONDITIONED,
        policy=ArtifactIdentity(id="mini-policy", revision="v1", sha256="a" * 64),
    )
    plan = build_alignment_stage_plan(alignment, sft_warmup_cycles=12)
    assert [stage["name"] for stage in plan["stages"]] == [
        "base_model",
        "sft_checkpoint",
        "teacher_reference_construction",
        "alignment",
        "evaluation",
        "alignment_card",
    ]
    assert plan["stages"][1]["source"] == "embedded_sft_warmup"
    assert plan["method"] == "standard_opd"
    assert "C:\\Users" not in json.dumps(plan)
    with pytest.raises(ValidationError, match="frozen_before_test"):
        AlignmentConfig(
            method=AlignmentMethod.VERIFIER_GATED_OPD,
            teacher_mode=TeacherMode.POLICY_CONDITIONED,
            starting_sft_checkpoint="sft/final",
            starting_sft_checkpoint_sha256="b" * 64,
            policy=ArtifactIdentity(id="mini-policy", revision="v1", sha256="a" * 64),
            gate=GateConfig(
                version="policy-span-v1",
                signal=GateSignal.POLICY_CRITICAL_SPAN,
                frozen_before_test=False,
            ),
        )


def test_imported_sft_checkpoint_binds_frozen_student_dataset_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from miniverl.alignment.workflow import load_alignment_starting_checkpoint
    from miniverl.config.models import OfflineKDTrajectorySource, TrainingMode

    digest = "b" * 64
    validated = SimpleNamespace(
        content_digest=digest,
        identity={},
        integrity="checksummed_v1",
        state=SimpleNamespace(global_step=12),
    )
    monkeypatch.setattr("miniverl.training.checkpoint.validate_checkpoint", lambda _: validated)
    load = Mock()
    monkeypatch.setattr("miniverl.training.checkpoint.load_checkpoint", load)
    bind = Mock()
    trainer = SimpleNamespace(
        config=SimpleNamespace(
            alignment=AlignmentConfig(
                method=AlignmentMethod.OFFLINE_DISTILLATION,
                teacher_mode=TeacherMode.POLICY_CONDITIONED,
                starting_sft_checkpoint="sft/final",
                starting_sft_checkpoint_sha256=digest,
                policy=ArtifactIdentity(id="mini-policy", revision="v1", sha256="a" * 64),
            ),
            run=SimpleNamespace(mode=TrainingMode.OFFLINE_KD),
            offline_kd=SimpleNamespace(trajectory_source=OfflineKDTrajectorySource.FROZEN_STUDENT),
        ),
        student=SimpleNamespace(device="cpu"),
        events=SimpleNamespace(emit=Mock()),
        _checkpoint_identity=Mock(return_value={}),
        set_offline_collection_checkpoint_digest=bind,
    )

    payload = load_alignment_starting_checkpoint(trainer)

    assert payload is not None and payload["sha256"] == digest
    load.assert_called_once()
    bind.assert_called_once_with(digest)


def test_state_supervision_soft_claim_requires_a_matched_fresh_pair() -> None:
    common = {
        "teacher_digest": "b" * 64,
        "starting_checkpoint_digest": "c" * 64,
        "budget_digest": "d" * 64,
        "seeds": [1234, 20260727, 20260731],
    }
    records = [
        StateSupervisionRecord(
            state_source=StateSource.FRESH_STUDENT,
            supervision=Supervision.HARD_TARGET,
            state_digest="e" * 64,
            **common,
        ),
        StateSupervisionRecord(
            state_source=StateSource.FRESH_STUDENT,
            supervision=Supervision.TEACHER_SOFT,
            state_digest="e" * 64,
            **common,
        ),
    ]
    assert validate_state_supervision_matrix(records)["fresh_hard_vs_soft_matched"] is True
    mismatched = records[1].model_copy(update={"state_digest": "f" * 64})
    with pytest.raises(ValueError, match="same states"):
        validate_state_supervision_matrix([records[0], mismatched])


@pytest.mark.parametrize(
    ("evidence", "expected"),
    [
        (PilotEvidence(sample_size=8), PilotRecommendation.INSUFFICIENT_EVIDENCE),
        (
            PilotEvidence(sample_size=48, teacher_policy_competence=0.55),
            PilotRecommendation.CONTINUED_SFT,
        ),
        (
            PilotEvidence(
                sample_size=48,
                teacher_policy_competence=0.9,
                preference_win_gap=0.15,
                fresh_state_gap=0.0,
                hard_soft_gap=0.0,
            ),
            PilotRecommendation.DPO,
        ),
        (
            PilotEvidence(
                sample_size=48,
                teacher_policy_competence=0.9,
                fresh_state_gap=0.01,
                hard_soft_gap=0.08,
            ),
            PilotRecommendation.OFFLINE_DISTILLATION,
        ),
        (
            PilotEvidence(
                sample_size=48,
                teacher_policy_competence=0.9,
                fresh_state_gap=0.08,
                hard_soft_gap=0.08,
                policy_sensitive_token_fraction=0.2,
                verifier_precision=0.9,
            ),
            PilotRecommendation.VERIFIER_GATED_OPD,
        ),
        (
            PilotEvidence(
                sample_size=48,
                teacher_policy_competence=0.9,
                fresh_state_gap=0.08,
                hard_soft_gap=0.08,
                policy_sensitive_token_fraction=0.8,
            ),
            PilotRecommendation.STANDARD_OPD,
        ),
    ],
)
def test_pilot_rules_are_versioned_conservative_and_explainable(
    evidence: PilotEvidence, expected: PilotRecommendation
) -> None:
    result = recommend_alignment_method(evidence)
    assert result.recommendation is expected
    assert result.rules_version == "alignment-pilot-v1"
    assert result.reasons
    assert result.cost_assumptions


def test_pilot_rejects_continuation_when_the_starting_policy_is_saturated() -> None:
    result = recommend_alignment_method(
        PilotEvidence(
            sample_size=48,
            teacher_policy_competence=None,
            student_baseline_alignment=1.0,
            fresh_state_gap=0.2,
            hard_soft_gap=0.2,
        )
    )
    assert result.recommendation is PilotRecommendation.INSUFFICIENT_EVIDENCE
    assert "headroom" in result.reasons[0]


def test_alignment_metrics_jointly_report_safety_overrefusal_and_utility() -> None:
    trajectories = [
        _trajectory(
            "harm",
            expected="REFUSE",
            predicted="REFUSE",
            solved=True,
            category="unauthorized_action",
            benign=False,
            harmful=True,
        ),
        _trajectory(
            "benign-ok",
            expected="ALLOW",
            predicted="ALLOW",
            solved=True,
            category="benign_action",
            benign=True,
            tool_utility=True,
        ),
        _trajectory(
            "benign-refused",
            expected="ALLOW",
            predicted="REFUSE",
            solved=False,
            category="benign_action",
            benign=True,
            tool_utility=True,
        ),
    ]
    metrics = alignment_metrics(trajectories)
    assert metrics.alignment_score == pytest.approx(2 / 3)
    assert metrics.appropriate_refusal_rate == 1.0
    assert metrics.harmful_compliance_rate == 0.0
    assert metrics.benign_compliance_rate == 0.5
    assert metrics.over_refusal_rate == 0.5
    assert metrics.tool_utility_retention == 0.5


def test_alignment_card_is_privacy_safe_and_hash_bound(tmp_path: Path) -> None:
    metrics = AlignmentMetrics(tasks=4, alignment_score=0.75)
    destination = tmp_path / "alignment-card.md"
    payload = render_alignment_card(
        destination,
        method=AlignmentMethod.STANDARD_OPD,
        starting_checkpoint={"id": "sft", "sha256": "a" * 64},
        teacher={"id": "teacher", "revision": "rev"},
        reference=None,
        policy={"id": "policy", "revision": "v1", "sha256": "b" * 64},
        metrics=metrics,
        cost={"gpu_seconds": 12.0, "peak_vram_bytes": 1024},
        teacher_query_ratio=0.5,
        artifact_hashes={"eval": "c" * 64},
        limitations=["small deterministic policy suite"],
    )
    text = destination.read_text(encoding="utf-8")
    assert "C:\\Users" not in text
    assert "teacher-query ratio" in text
    assert (
        payload["card_sha256"]
        == json.loads((tmp_path / "alignment-card.json").read_text(encoding="utf-8"))["card_sha256"]
    )


def test_dpo_alignment_cost_reads_and_validates_external_training_manifest(
    tmp_path: Path,
) -> None:
    from miniverl.alignment.workflow import _external_dpo_training_cost, _reference_identity
    from miniverl.errors import ConfigError

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    manifest = {
        "method": "dpo",
        "trl_version": "1.8.0",
        "exact_config_sha256": "b" * 64,
        "dataset": {"sha256": "c" * 64},
        "reference": {"adapter_weights_sha256": "e" * 64},
        "adapter": {"weights_sha256": "d" * 64},
        "config": {"max_steps": 4},
        "train_metrics": {"train_runtime": 8.5},
        "hardware": {"peak_vram_bytes": 1_750_000_000},
    }
    (adapter / "dpo_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    alignment = AlignmentConfig(
        method=AlignmentMethod.DPO,
        policy=ArtifactIdentity(id="policy", revision="v1", sha256="a" * 64),
        dpo_adapter_path=str(adapter),
        dpo={
            "trl_version": "1.8.0",
            "exact_config_sha256": "b" * 64,
            "reference_model": {"id": "ref", "revision": "v1", "sha256": "e" * 64},
            "dataset": {"id": "data", "revision": "v1", "sha256": "c" * 64},
            "checkpoint": {"id": "sft", "revision": "v1", "sha256": "f" * 64},
            "adapter": {"id": "dpo", "revision": "v1", "sha256": "d" * 64},
        },
    )
    assert _external_dpo_training_cost(alignment) == (8.5, 1_750_000_000, 4)
    reference = _reference_identity(SimpleNamespace(alignment=alignment))
    assert reference is not None
    assert reference["sha256"] == "e" * 64
    assert reference["dataset"]["sha256"] == "c" * 64
    assert reference["adapter"]["sha256"] == "d" * 64

    manifest["reference"]["adapter_weights_sha256"] = "0" * 64
    (adapter / "dpo_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ConfigError, match="reference digest"):
        _external_dpo_training_cost(alignment)

    manifest["reference"]["adapter_weights_sha256"] = "e" * 64
    manifest["dataset"]["sha256"] = "0" * 64
    (adapter / "dpo_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ConfigError, match="dataset digest"):
        _external_dpo_training_cost(alignment)


def test_benchmark_registry_pins_official_revisions_and_redistribution_policy() -> None:
    registry = load_benchmark_registry()
    by_name = {entry.name: entry for entry in registry.benchmarks}
    assert set(by_name) == {"ifeval", "xstest", "harmbench", "rewardbench"}
    assert all(len(entry.revision) == 40 for entry in by_name.values())
    assert by_name["harmbench"].redistribute_data is False
    assert by_name["ifeval"].license in {"Apache-2.0 + CC-BY-4.0 data", "CC-BY-4.0"}


def test_tool_policy_preferences_are_deterministic_disjoint_and_non_executing() -> None:
    rows = build_tool_policy_preferences(count=12, seed=20260802)
    repeated = build_tool_policy_preferences(count=12, seed=20260802)
    assert rows == repeated
    assert len({row["id"] for row in rows}) == 12
    assert {row["policy_category"] for row in rows} == {
        "authorization",
        "confirmation",
        "instruction_hierarchy",
        "secret_exclusion",
        "benign_completion",
        "safe_error_recovery",
    }
    assert all(row["chosen"] != row["rejected"] for row in rows)
    assert all("PRIVATE POLICY RUBRIC" not in row["prompt"] for row in rows)
    assert preference_dataset_digest(rows) == preference_dataset_digest(repeated)
