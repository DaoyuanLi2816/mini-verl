#!/usr/bin/env python3
"""Audit and render the preregistered Alignment Lab v1 final artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from miniverl.alignment.evaluation import alignment_metrics
from miniverl.alignment.pilot import recommend_alignment_method
from miniverl.alignment.schema import PilotEvidence
from miniverl.schemas.trajectory import Trajectory
from miniverl.utils.runs import canonical_json

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "benchmarks/results/alignment-lab-v1.json"
TASK_RESULTS = ROOT / "benchmarks/results/alignment-lab-v1-task-results.jsonl"
STATE_SUPERVISION = ROOT / "benchmarks/results/alignment-lab-v1-state-supervision.json"
PREREGISTRATION = ROOT / "benchmarks/preregistration/alignment-lab-v1.yaml"
DOCS = ROOT / "docs/alignment-lab"
CARDS = ROOT / "benchmarks/alignment-cards/alignment-lab-v1"
PILOT_EXAMPLE = ROOT / "examples/alignment-lab/pilot.json"
FROZEN_CALCULATOR = ROOT / "benchmarks/results/gpu-calc-hard-equal-update-v2.json"
FROZEN_CALCULATOR_SHA256 = "53fc1d4d5b7adee09618d77ad62d4086ba56b78569832d6fc7c3bcd5c2695bbc"
PREREGISTRATION_SHA256 = "71307dbfe9a5bb20c686307cafce8bd254c07af8b69c1bf1c6ec0dbf53a8cde0"
STARTING_CHECKPOINT_SHA256 = "7304922281268a687dd1c75ba918e26c64c8207b5701db78c368afd20d80ae89"
POLICY_SHA256 = "9a9316bea117928d115eff86291982d7386e6ca2d7127aacb933e508d322c8a8"
CODE_COMMIT = "8ffb2d9c3094fb6580e5afd89c861c8e0511ef78"
METHODS = (
    "sft_checkpoint",
    "continued_sft",
    "dpo",
    "offline_distillation",
    "standard_opd",
    "verifier_gated_opd",
)
SEEDS = (1234, 20260727, 20260801)
LABELS = {
    "sft_checkpoint": "SFT checkpoint",
    "continued_sft": "continued SFT",
    "dpo": "DPO",
    "offline_distillation": "offline soft distillation",
    "standard_opd": "standard OPD",
    "verifier_gated_opd": "verifier-gated OPD",
}
COLORS = {
    "sft_checkpoint": "#A7A9AC",
    "continued_sft": "#0072B2",
    "dpo": "#CC79A7",
    "offline_distillation": "#E69F00",
    "standard_opd": "#D55E00",
    "verifier_gated_opd": "#009E73",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _json_line(payload: dict[str, Any]) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def _final_trajectories(run: Path) -> list[Trajectory]:
    rows = [Trajectory.model_validate(row) for row in _jsonl(run / "eval_trajectories.jsonl")]
    return [row for row in rows if ":final:v" in row.trajectory_id]


def _artifact_hashes(run: Path) -> dict[str, str]:
    names = (
        "alignment-card.json",
        "alignment.json",
        "config.resolved.yaml",
        "eval.json",
        "eval_trajectories.jsonl",
        "events.jsonl",
        "manifest.json",
        "metrics.jsonl",
    )
    return {name: _sha256(run / name) for name in names}


def _sanitize_task(method: str, seed: int, row: Trajectory) -> dict[str, Any]:
    verification = row.verification
    if verification is None:
        raise ValueError(f"trajectory has no verification: {row.trajectory_id}")
    metadata = row.metadata
    response_tokens = [
        token
        for token, generated in zip(row.token_ids, row.model_generated_mask, strict=True)
        if generated
    ]
    return {
        "schema_version": 1,
        "method": method,
        "seed": seed,
        "task_id": row.task_id,
        "policy_category": str(metadata.get("policy_category")),
        "expected_decision": str(metadata.get("expected_decision")),
        "predicted_decision": str(verification.predicted or ""),
        "solved": bool(verification.solved),
        "benign": bool(metadata.get("benign")),
        "harmful": bool(metadata.get("harmful")),
        "tool_utility": bool(metadata.get("tool_utility")),
        "instruction_retention": bool(metadata.get("instruction_retention")),
        "response_token_sha256": hashlib.sha256(
            canonical_json(response_tokens).encode("utf-8")
        ).hexdigest(),
    }


def _run_card(run: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = _json(run / "manifest.json")
    alignment = _json(run / "alignment.json")
    if manifest.get("status") != "completed":
        raise ValueError(f"run is not completed: {run.name}")
    if not manifest.get("all_expected_artifacts_complete"):
        raise ValueError(f"run has incomplete artifacts: {run.name}")
    return manifest, alignment["card"]


def _method_run(root: Path, method: str, seed: int) -> Path:
    return root / f"test-{method}-seed-{seed}"


def _metrics(rows: list[Trajectory], card: dict[str, Any]) -> dict[str, Any]:
    measured = alignment_metrics(rows).model_dump(mode="json")
    card_metrics = card["metrics"]
    for key in (
        "teacher_queried_positions",
        "teacher_query_ratio",
        "decision_distribution_shift_jsd",
    ):
        measured[key] = card_metrics.get(key)
    measured.pop("gpu_seconds", None)
    measured.pop("peak_vram_bytes", None)
    return measured


def _dpo_manifest(dpo_root: Path, seed: int) -> tuple[Path, dict[str, Any]]:
    path = dpo_root / f"v05-alignment-dpo-seed-{seed}" / "dpo_manifest.json"
    payload = _json(path)
    if payload.get("method") != "dpo" or payload.get("seed") != seed:
        raise ValueError(f"invalid DPO manifest identity: {path}")
    return path, payload


def _arm(
    *,
    method: str,
    seed: int,
    runs: list[tuple[Path, int]],
    dpo_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    trajectories: list[Trajectory] = []
    manifests: list[dict[str, Any]] = []
    cards: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
    artifacts: dict[str, Any] = {}
    for run, task_offset in runs:
        manifest, card = _run_card(run)
        rows = _final_trajectories(run)
        expected_count = int(manifest["result"]["eval"]["tasks"])
        if len(rows) != expected_count:
            raise ValueError(f"trajectory count mismatch: {run.name}")
        if int(manifest["result"]["eval"].get("task_offset") or 0) != task_offset:
            raise ValueError(f"task offset mismatch: {run.name}")
        trajectories.extend(rows)
        manifests.append(manifest)
        cards.append(card)
        key = "full" if len(runs) == 1 else f"offset_{task_offset}"
        artifacts[key] = _artifact_hashes(run)
        segments.append(
            {
                "task_offset": task_offset,
                "tasks": len(rows),
                "code_commit": manifest["git_commit"],
                "artifact_tree_sha256": hashlib.sha256(
                    canonical_json(artifacts[key]).encode("utf-8")
                ).hexdigest(),
            }
        )
    if len(trajectories) != 48 or len({row.task_id for row in trajectories}) != 48:
        raise ValueError(f"arm is not 48 unique paired tasks: {method} seed {seed}")
    if any(
        card["starting_sft_checkpoint"].get("sha256") != STARTING_CHECKPOINT_SHA256
        for card in cards
    ):
        raise ValueError(f"starting checkpoint drift: {method} seed {seed}")

    gpu_seconds = sum(float(card["cost"].get("gpu_seconds") or 0.0) for card in cards)
    wall_seconds = sum(float(card["cost"].get("wall_seconds") or 0.0) for card in cards)
    peak_vram = max(int(card["cost"].get("peak_vram_bytes") or 0) for card in cards)
    optimizer_updates = sum(int(card["cost"].get("optimizer_updates") or 0) for card in cards)
    provenance: dict[str, Any] = {
        "run_code_commits": sorted({manifest["git_commit"] for manifest in manifests}),
        "resolved_config_sha256": [
            manifest["config_digests"]["resolved"] for manifest in manifests
        ],
    }
    if method in {"standard_opd", "verifier_gated_opd"}:
        events = _jsonl(runs[0][0] / "events.jsonl")
        rollouts = [row for row in events if row.get("event") == "rollouts_collected"]
        if len(rollouts) != 4 or any(
            row.get("rollout_policy_version") != row.get("parameter_version") for row in rollouts
        ):
            raise ValueError(f"strict OPD freshness audit failed: {method} seed {seed}")
        provenance["freshness_audit"] = {
            "updates": 4,
            "rollout_policy_equals_current_parameter_version": True,
        }
        if method == "verifier_gated_opd":
            decisions = [row for row in events if row.get("event") == "alignment_gate_decision"]
            if len(decisions) != 16 or any(
                row.get("gate_version") != "policy-critical-span-v1" for row in decisions
            ):
                raise ValueError(f"verifier-gate audit failed: seed {seed}")
            provenance["gate_audit"] = {
                "version": "policy-critical-span-v1",
                "example_decisions": 16,
                "all_spans_recorded": all(isinstance(row.get("spans"), list) for row in decisions),
            }
    if method == "offline_distillation":
        dataset = manifests[0].get("offline_dataset") or {}
        digest = dataset.get("digest")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"offline dataset provenance is incomplete: seed {seed}")
        provenance["frozen_student_dataset_sha256"] = digest
    if method == "dpo":
        dpo_path, dpo = _dpo_manifest(dpo_root, seed)
        train_seconds = float(dpo["train_metrics"]["train_runtime"])
        gpu_seconds += train_seconds
        wall_seconds += train_seconds
        peak_vram = max(peak_vram, int(dpo["hardware"]["peak_vram_bytes"]))
        optimizer_updates = int(dpo["config"]["max_steps"])
        dpo_identity = {
            "trl_version": dpo["trl_version"],
            "exact_config_sha256": dpo["exact_config_sha256"],
            "base_model": dpo["base_model"],
            "reference": dpo["reference"],
            "dataset": dpo["dataset"],
            "checkpoint": {
                "id": "common-sft-checkpoint",
                "sha256": STARTING_CHECKPOINT_SHA256,
            },
            "adapter": {
                "id": dpo_path.parent.name,
                **dpo["adapter"],
            },
            "manifest_sha256": _sha256(dpo_path),
        }
        provenance.update(
            {
                "trl_version": dpo["trl_version"],
                "exact_config_sha256": dpo["exact_config_sha256"],
                "preference_dataset_sha256": dpo["dataset"]["sha256"],
                "adapter_weights_sha256": dpo["adapter"]["weights_sha256"],
                "dpo_manifest_sha256": _sha256(dpo_path),
                "external_training_included_in_cost": True,
                "dpo_identity": dpo_identity,
            }
        )
    failed_rows = [row for row in trajectories if row.verification and not row.verification.solved]
    failure_categories: dict[str, int] = {}
    policy_categories: dict[str, int] = {}
    for row in failed_rows:
        assert row.verification is not None
        failure = str(row.verification.failure_category or "unclassified")
        category = str(row.metadata.get("policy_category") or "unclassified")
        failure_categories[failure] = failure_categories.get(failure, 0) + 1
        policy_categories[category] = policy_categories.get(category, 0) + 1
    provenance["final_failure_audit"] = {
        "failed_tasks": len(failed_rows),
        "failure_categories": dict(sorted(failure_categories.items())),
        "policy_categories": dict(sorted(policy_categories.items())),
        "task_ids": sorted(row.task_id for row in failed_rows),
    }
    recovery: dict[str, Any] | None = None
    if len(runs) == 2:
        recovery = {
            "kind": "two_disjoint_segments",
            "reason": "preregistration_revision_1.4_split_capacity_recovery",
            "segments": segments,
            "repeated_test_tasks": 0,
        }
    arm = {
        "method": method,
        "seed": seed,
        "status": "completed",
        "starting_checkpoint_sha256": STARTING_CHECKPOINT_SHA256,
        "metrics": _metrics(trajectories, cards[-1]),
        "cost": {
            "gpu_seconds": round(gpu_seconds, 6),
            "wall_seconds": round(wall_seconds, 6),
            "peak_vram_bytes": peak_vram,
            "optimizer_updates": optimizer_updates,
        },
        "artifacts": artifacts,
        "provenance": provenance,
        "recovery": recovery,
    }
    task_rows = [_sanitize_task(method, seed, row) for row in trajectories]
    return arm, task_rows


def _mean(values: list[float | int | None]) -> float | None:
    actual = [float(value) for value in values if value is not None]
    return statistics.fmean(actual) if actual else None


def _summary(arms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for method in METHODS:
        rows = [arm for arm in arms if arm["method"] == method]
        output.append(
            {
                "method": method,
                "label": LABELS[method],
                "seeds": [row["seed"] for row in rows],
                "alignment_score_mean": _mean([row["metrics"]["alignment_score"] for row in rows]),
                "appropriate_refusal_rate_mean": _mean(
                    [row["metrics"]["appropriate_refusal_rate"] for row in rows]
                ),
                "harmful_compliance_rate_mean": _mean(
                    [row["metrics"]["harmful_compliance_rate"] for row in rows]
                ),
                "over_refusal_rate_mean": _mean(
                    [row["metrics"]["over_refusal_rate"] for row in rows]
                ),
                "benign_compliance_rate_mean": _mean(
                    [row["metrics"]["benign_compliance_rate"] for row in rows]
                ),
                "preference_win_rate_mean": _mean(
                    [row["metrics"]["preference_win_rate"] for row in rows]
                ),
                "instruction_retention_mean": _mean(
                    [row["metrics"]["instruction_retention"] for row in rows]
                ),
                "tool_utility_retention_mean": _mean(
                    [row["metrics"]["tool_utility_retention"] for row in rows]
                ),
                "general_utility_retention_mean": _mean(
                    [row["metrics"]["general_utility_retention"] for row in rows]
                ),
                "teacher_query_ratio_mean": _mean(
                    [row["metrics"]["teacher_query_ratio"] for row in rows]
                ),
                "gpu_seconds_mean": _mean([row["cost"]["gpu_seconds"] for row in rows]),
                "peak_vram_bytes_max": max(row["cost"]["peak_vram_bytes"] for row in rows),
            }
        )
    return output


def _state_supervision() -> dict[str, Any]:
    payload = _json(STATE_SUPERVISION)
    if payload.get("name") != "alignment-lab-v1-state-supervision":
        raise ValueError("not the Alignment Lab v1 State x Supervision diagnostic")
    if payload.get("measurement_status") != "measured_signal_diagnostic_not_training_outcome":
        raise ValueError("State x Supervision artifact has the wrong measurement status")
    comparisons = payload.get("matched_comparisons") or {}
    if set(comparisons) != {
        "frozen_hard_vs_fresh_hard",
        "frozen_soft_vs_fresh_soft",
        "fresh_hard_vs_fresh_soft",
    }:
        raise ValueError("State x Supervision artifact is missing a required comparison")
    return {
        "artifact": STATE_SUPERVISION.name,
        "sha256": _sha256(STATE_SUPERVISION),
        "measurement_status": payload["measurement_status"],
        "hard_definition": payload["hard_definition"],
        "matched_comparisons": comparisons,
        "headline_training_cells": payload["headline_training_cells"],
        "claims": payload["claims"],
    }


def _pilot(summary: list[dict[str, Any]], state_supervision: dict[str, Any]) -> dict[str, Any]:
    online = next(row for row in summary if row["method"] == "standard_opd")
    offline = next(row for row in summary if row["method"] == "offline_distillation")
    baseline = next(row for row in summary if row["method"] == "sft_checkpoint")
    fresh_gap = float(online["alignment_score_mean"]) - float(offline["alignment_score_mean"])
    hard_soft = state_supervision["matched_comparisons"]["fresh_hard_vs_fresh_soft"]
    gated = next(row for row in summary if row["method"] == "verifier_gated_opd")
    evidence = PilotEvidence(
        sample_size=48,
        teacher_policy_competence=None,
        student_baseline_alignment=float(baseline["alignment_score_mean"]),
        teacher_student_policy_gap=None,
        teacher_student_topk_overlap=None,
        fresh_state_gap=fresh_gap,
        hard_soft_gap=float(hard_soft["soft_probability_mass_beyond_argmax_mean"]),
        preference_win_gap=float(online["preference_win_rate_mean"])
        - float(baseline["preference_win_rate_mean"]),
        policy_sensitive_token_fraction=float(gated["teacher_query_ratio_mean"]),
        verifier_precision=None,
        estimated_vram_gib=float(online["peak_vram_bytes_max"]) / 2**30,
        estimated_time_seconds=float(online["gpu_seconds_mean"]),
        uncertainty_half_width=None,
    )
    result = recommend_alignment_method(evidence).model_dump(mode="json")
    return {
        **result,
        "scope": "alignment-lab-v1 measured final; three seeds; one deterministic suite",
        "evidence_status": {
            "teacher_policy_competence": "not_measured_as_a_free_running_teacher_endpoint",
            "teacher_student_policy_gap": "not_computable_without_teacher_policy_competence",
            "teacher_student_topk_overlap": "student_distribution_not_retained",
            "verifier_precision": "gate_coverage_measured_but_precision_not_independently_estimated",
            "uncertainty_half_width": "not_a_population_sample",
        },
        "decision": "Do not spend online teacher-query cost on this already-saturated recipe.",
    }


def publish(
    *,
    run_root: Path,
    baseline_prefix: Path,
    baseline_recovery: Path,
    dpo_root: Path,
    result_path: Path = RESULT,
    task_path: Path = TASK_RESULTS,
) -> dict[str, str]:
    if _sha256(PREREGISTRATION) != PREREGISTRATION_SHA256:
        raise ValueError("Alignment Lab preregistration digest changed")
    if _sha256(FROZEN_CALCULATOR) != FROZEN_CALCULATOR_SHA256:
        raise ValueError("immutable calculator benchmark hash changed")
    arms: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    expected_task_ids: tuple[str, ...] | None = None
    for seed in SEEDS:
        for method in METHODS:
            if seed == 1234 and method == "sft_checkpoint":
                runs = [(baseline_prefix, 0), (baseline_recovery, 24)]
            else:
                runs = [(_method_run(run_root, method, seed), 0)]
            arm, rows = _arm(method=method, seed=seed, runs=runs, dpo_root=dpo_root)
            ordered = tuple(row["task_id"] for row in rows)
            if expected_task_ids is None:
                expected_task_ids = ordered
            elif ordered != expected_task_ids:
                raise ValueError(f"ordered final tasks are not paired: {method} seed {seed}")
            arms.append(arm)
            task_rows.extend(rows)
    task_text = "".join(_json_line(row) + "\n" for row in task_rows)
    task_path.parent.mkdir(parents=True, exist_ok=True)
    task_path.write_text(task_text, encoding="utf-8", newline="\n")
    summary = _summary(arms)
    state_supervision = _state_supervision()
    pilot = _pilot(summary, state_supervision)
    payload = {
        "schema_version": 1,
        "name": "alignment-lab-v1",
        "measurement_status": "measured_final",
        "code_commit": CODE_COMMIT,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "frozen_calculator_sha256": FROZEN_CALCULATOR_SHA256,
        "starting_checkpoint_sha256": STARTING_CHECKPOINT_SHA256,
        "policy_sha256": POLICY_SHA256,
        "hardware": {
            "gpu": "NVIDIA GeForce RTX 4080",
            "total_vram_bytes": 17170956288,
            "torch": "2.13.0+cu130",
            "transformers": "5.14.1",
            "peft": "0.19.1",
            "cross_gpu_generalization": "not_tested",
        },
        "scope": {
            "model": "Qwen/Qwen3-0.6B",
            "model_revision": "c1899de289a04d12100db370d81485cdf75e47ca",
            "suite": "Minipolicy v1 deterministic tool-policy sandbox",
            "test_tasks_per_arm": 48,
            "seeds": list(SEEDS),
            "ordinary_capability": "saturated_before_continuation",
            "external_alignment_benchmarks": "metadata_adapters_only_not_measured",
            "claim_limit": (
                "Paired observation on one model, deterministic suite and GPU; no broad safety, "
                "capability, population or cross-hardware claim."
            ),
        },
        "arms": arms,
        "method_summary": summary,
        "state_supervision_diagnostic": state_supervision,
        "pilot": pilot,
        "task_results_sha256": _sha256(task_path),
        "invalidation_status": {"valid": True, "reasons": []},
    }
    result_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    source_digest = _sha256(result_path)
    DOCS.mkdir(parents=True, exist_ok=True)
    for name, content in render_figures(payload, source_digest).items():
        (DOCS / name).write_text(content, encoding="utf-8", newline="\n")
    report = render_report(payload, source_digest)
    (DOCS / "alignment-lab-v1.md").write_text(report, encoding="utf-8", newline="\n")
    CARDS.mkdir(parents=True, exist_ok=True)
    for name, content in render_cards(payload, source_digest).items():
        (CARDS / name).write_text(content, encoding="utf-8", newline="\n")
    PILOT_EXAMPLE.parent.mkdir(parents=True, exist_ok=True)
    PILOT_EXAMPLE.write_text(
        json.dumps(pilot, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return {
        result_path.relative_to(ROOT).as_posix(): source_digest,
        task_path.relative_to(ROOT).as_posix(): _sha256(task_path),
        **{
            (DOCS / name).relative_to(ROOT).as_posix(): _sha256(DOCS / name)
            for name in render_figures(payload, source_digest)
        },
        **{
            (CARDS / name).relative_to(ROOT).as_posix(): _sha256(CARDS / name)
            for name in render_cards(payload, source_digest)
        },
        (DOCS / "alignment-lab-v1.md").relative_to(ROOT).as_posix(): _sha256(
            DOCS / "alignment-lab-v1.md"
        ),
        PILOT_EXAMPLE.relative_to(ROOT).as_posix(): _sha256(PILOT_EXAMPLE),
    }


def _load_result(path: Path) -> dict[str, Any]:
    payload = _json(path)
    if payload.get("schema_version") != 1 or payload.get("name") != "alignment-lab-v1":
        raise ValueError("not an Alignment Lab v1 result")
    if payload.get("measurement_status") != "measured_final":
        raise ValueError("Alignment Lab result is not a final measurement")
    arms = payload.get("arms") or []
    keys = {(arm.get("method"), arm.get("seed")) for arm in arms}
    if keys != {(method, seed) for seed in SEEDS for method in METHODS}:
        raise ValueError("Alignment Lab final matrix is incomplete")
    if any(arm.get("status") != "completed" for arm in arms):
        raise ValueError("Alignment Lab contains an incomplete arm")
    if any((arm.get("metrics") or {}).get("tasks") != 48 for arm in arms):
        raise ValueError("Alignment Lab final arm does not contain 48 tasks")
    if payload.get("preregistration_sha256") != PREREGISTRATION_SHA256:
        raise ValueError("Alignment Lab result has the wrong preregistration digest")
    if payload.get("frozen_calculator_sha256") != FROZEN_CALCULATOR_SHA256:
        raise ValueError("Alignment Lab result has the wrong calculator digest")
    if _sha256(FROZEN_CALCULATOR) != FROZEN_CALCULATOR_SHA256:
        raise ValueError("immutable calculator benchmark hash changed")
    return payload


def _svg_shell(
    title: str, description: str, subtitle: str, body: list[str], *, height: int = 720
) -> str:
    style = (
        "text{font-family:'DejaVu Sans','Segoe UI',sans-serif;fill:#edf4ff}"
        ".title{font-size:31px;font-weight:760}.sub{font-size:17px;fill:#aebbd2}"
        ".axis{font-size:17px;fill:#aebbd2}.label{font-size:18px;font-weight:650}"
        ".value{font-size:17px;font-weight:760}.small{font-size:16px;fill:#b9c5d8}"
        ".header{font-size:17px;font-weight:700;fill:#dce7f8}"
    )
    return "".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="{height}" '
            f'viewBox="0 0 1120 {height}" role="img" aria-labelledby="title desc">',
            f'<title id="title">{escape(title)}</title>',
            f'<desc id="desc">{escape(description)}</desc>',
            f'<rect width="1120" height="{height}" rx="24" fill="#060a14"/>',
            f'<rect x="20" y="20" width="1080" height="{height - 40}" rx="20" '
            'fill="#0a1222" stroke="#20304f"/>',
            f"<style>{style}</style>",
            f'<text class="title" x="48" y="66">{escape(title)}</text>',
            f'<text class="sub" x="48" y="98">{escape(subtitle)}</text>',
            *body,
            "</svg>\n",
        ]
    )


def _scale(value: float, *, low: float, high: float, start: float, end: float) -> float:
    if value < low - 1e-9 or value > high + 1e-9:
        raise ValueError(f"quantitative value {value} lies outside [{low}, {high}]")
    return start + (value - low) * (end - start) / (high - low)


def _seed_mark(*, x: float, y: float, method: str, seed: int, value: float) -> str:
    color = COLORS[method]
    common = (
        f'data-encoding="seed-point" data-seed="{seed}" data-value="{value:.10g}" '
        f'fill="{color}" stroke="#07111f" stroke-width="1.5"'
    )
    if seed == SEEDS[0]:
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" {common}/>'
    if seed == SEEDS[1]:
        points = f"{x:.1f},{y - 5:.1f} {x + 5:.1f},{y:.1f} {x:.1f},{y + 5:.1f} {x - 5:.1f},{y:.1f}"
        return f'<polygon points="{points}" {common}/>'
    return (
        f'<path d="M{x - 4.5:.1f},{y - 4.5:.1f} L{x + 4.5:.1f},{y + 4.5:.1f} '
        f'M{x + 4.5:.1f},{y - 4.5:.1f} L{x - 4.5:.1f},{y + 4.5:.1f}" '
        f'data-encoding="seed-point" data-seed="{seed}" data-value="{value:.10g}" '
        f'fill="none" stroke="{color}" stroke-width="3"/>'
    )


def _method_arms(payload: dict[str, Any], method: str) -> list[dict[str, Any]]:
    rows = sorted(
        (arm for arm in payload["arms"] if arm["method"] == method),
        key=lambda arm: SEEDS.index(int(arm["seed"])),
    )
    if [int(row["seed"]) for row in rows] != list(SEEDS):
        raise ValueError(f"method {method} does not contain the three measured seeds")
    return rows


def _pp(value: float) -> str:
    return "0.0" if math.isclose(value, 0.0, abs_tol=5e-5) else f"{value:+.1f}"


def _delta_from_sft(payload: dict[str, Any]) -> str:
    body: list[str] = [
        '<text class="header" x="48" y="135">Continuation method</text>',
        '<text class="header" x="370" y="135">delta from the same-seed SFT checkpoint (percentage points)</text>',
        '<circle cx="904" cy="130" r="7" fill="#edf4ff"/>',
        '<text class="small" x="918" y="136">alignment mean</text>',
        '<rect x="1020" y="123" width="14" height="14" fill="#edf4ff"/>',
        '<text class="small" x="1040" y="136">utility</text>',
    ]
    left, right = 370.0, 1040.0
    low, high = -40.0, 5.0
    for tick in (-40, -30, -20, -10, 0, 5):
        x = _scale(float(tick), low=low, high=high, start=left, end=right)
        stroke = "#f4f7fb" if tick == 0 else "#233553"
        width = 2.5 if tick == 0 else 1
        body.extend(
            [
                f'<line x1="{x:.1f}" y1="151" x2="{x:.1f}" y2="625" stroke="{stroke}" stroke-width="{width}" data-axis-domain="-40,5"/>',
                f'<text class="axis" x="{x:.1f}" y="656" text-anchor="middle">{tick:+d}</text>',
            ]
        )
    body.append(
        f'<text class="small" x="{_scale(0, low=low, high=high, start=left, end=right) + 8:.1f}" y="174">zero baseline</text>'
    )
    baseline = {int(arm["seed"]): arm for arm in _method_arms(payload, "sft_checkpoint")}
    for index, method in enumerate(METHODS[1:]):
        y = 210 + index * 86
        arms = _method_arms(payload, method)
        alignment = [
            100
            * (
                float(arm["metrics"]["alignment_score"])
                - float(baseline[int(arm["seed"])]["metrics"]["alignment_score"])
            )
            for arm in arms
        ]
        utility = [
            100
            * (
                float(arm["metrics"]["tool_utility_retention"])
                - float(baseline[int(arm["seed"])]["metrics"]["tool_utility_retention"])
            )
            for arm in arms
        ]
        align_mean = statistics.fmean(alignment)
        utility_mean = statistics.fmean(utility)
        body.extend(
            [
                f'<text class="label" data-role="chart-label" x="48" y="{y - 14}">{escape(LABELS[method])}</text>',
                f'<text class="small" data-role="chart-label" x="48" y="{y + 11}">A seeds: {" / ".join(_pp(v) for v in alignment)}</text>',
                f'<text class="small" data-role="chart-label" x="48" y="{y + 34}">U seeds: {" / ".join(_pp(v) for v in utility)}</text>',
                f'<line x1="{left}" y1="{y}" x2="{right}" y2="{y}" stroke="#182a46"/>',
            ]
        )
        for arm, value in zip(arms, alignment, strict=True):
            body.append(
                _seed_mark(
                    x=_scale(value, low=low, high=high, start=left, end=right),
                    y=y - 11,
                    method=method,
                    seed=int(arm["seed"]),
                    value=value,
                )
            )
        for arm, value in zip(arms, utility, strict=True):
            body.append(
                _seed_mark(
                    x=_scale(value, low=low, high=high, start=left, end=right),
                    y=y + 11,
                    method=method,
                    seed=int(arm["seed"]),
                    value=value,
                )
            )
        ax = _scale(align_mean, low=low, high=high, start=left, end=right)
        ux = _scale(utility_mean, low=low, high=high, start=left, end=right)
        body.extend(
            [
                f'<circle cx="{ax:.1f}" cy="{y - 11}" r="8" fill="none" stroke="{COLORS[method]}" stroke-width="3" data-value="{align_mean:.10g}"/>',
                f'<rect x="{ux - 8:.1f}" y="{y + 3:.1f}" width="16" height="16" fill="none" stroke="{COLORS[method]}" stroke-width="3" data-value="{utility_mean:.10g}"/>',
                f'<text class="value" data-role="chart-label" x="{max(left + 8, min(ax - 12, right - 75)):.1f}" y="{y - 27}" text-anchor="end">A {_pp(align_mean)}</text>',
                f'<text class="value" data-role="chart-label" x="{max(left + 8, min(ux - 12, right - 75)):.1f}" y="{y + 29}" text-anchor="end">U {_pp(utility_mean)}</text>',
            ]
        )
    body.append(
        '<text class="small" x="48" y="688">Seed shapes: ● 1234 · ◆ 20260727 · × 20260801. A = alignment; U = retained tool utility.</text>'
    )
    return _svg_shell(
        "Continuation could not improve the saturated SFT checkpoint",
        (
            "Forest chart of alignment-score and retained-tool-utility percentage-point deltas "
            "for five continuation methods. Every seed is shown at its exact x value."
        ),
        "48 paired sandbox tasks per seed · mean marks plus all three measured seeds · zero = starting SFT",
        body,
    )


def _matrix_cell(
    *,
    payload: dict[str, Any],
    x: float,
    y: float,
    values: list[float],
    mean: float,
    domain_high: float,
    method: str,
    formatter: Any,
    main: float | None = None,
) -> list[str]:
    bar_width = 84.0
    main_value = mean if main is None else main
    main_x = _scale(main_value, low=0.0, high=domain_high, start=x, end=x + bar_width)
    body = [
        f'<rect x="{x:.1f}" y="{y - 9:.1f}" width="{bar_width}" height="18" rx="4" fill="#132541"/>',
        f'<rect x="{x:.1f}" y="{y - 9:.1f}" width="{max(0.0, main_x - x):.1f}" height="18" rx="4" fill="{COLORS[method]}" opacity="0.72" data-value="{main_value:.10g}" data-axis-domain="0,{domain_high:.10g}"/>',
    ]
    for arm, value in zip(_method_arms(payload, method), values, strict=True):
        body.append(
            _seed_mark(
                x=_scale(value, low=0.0, high=domain_high, start=x, end=x + bar_width),
                y=y,
                method=method,
                seed=int(arm["seed"]),
                value=value,
            )
        )
    body.append(
        f'<text class="value" data-role="chart-label" x="{x + 145:.1f}" y="{y + 6:.1f}" text-anchor="end">{escape(formatter(main_value))}</text>'
    )
    return body


def _outcome_cost_matrix(payload: dict[str, Any]) -> str:
    columns = (250.0, 415.0, 580.0, 745.0, 910.0)
    body: list[str] = [
        '<text class="header" x="48" y="142">Method</text>',
        '<text class="header" x="250" y="132">Alignment</text>',
        '<text class="small" x="250" y="154">0–100%</text>',
        '<text class="header" x="415" y="132">Tool utility</text>',
        '<text class="small" x="415" y="154">0–100%</text>',
        '<text class="header" x="580" y="132">Teacher query</text>',
        '<text class="small" x="580" y="154">0–100%</text>',
        '<text class="header" x="745" y="132">GPU time</text>',
        '<text class="small" x="745" y="154">0–100 seconds</text>',
        '<text class="header" x="910" y="132">Peak VRAM</text>',
        '<text class="small" x="910" y="154">0–2 GiB</text>',
    ]
    summary = {row["method"]: row for row in payload["method_summary"]}
    for index, method in enumerate(METHODS):
        y = 195 + index * 78
        arms = _method_arms(payload, method)
        row = summary[method]
        alignment = [100 * float(arm["metrics"]["alignment_score"]) for arm in arms]
        utility = [100 * float(arm["metrics"]["tool_utility_retention"]) for arm in arms]
        time_values = [float(arm["cost"]["gpu_seconds"]) for arm in arms]
        vram_values = [float(arm["cost"]["peak_vram_bytes"]) / 2**30 for arm in arms]
        body.extend(
            [
                f'<rect x="36" y="{y - 31}" width="1048" height="62" rx="10" fill="{("#0d192d" if index % 2 == 0 else "#0a1426")}"/>',
                f'<rect x="48" y="{y - 11}" width="12" height="22" rx="3" fill="{COLORS[method]}"/>',
                f'<text class="label" data-role="chart-label" x="70" y="{y + 6}">{escape(LABELS[method])}</text>',
            ]
        )
        body.extend(
            _matrix_cell(
                payload=payload,
                x=columns[0],
                y=y,
                values=alignment,
                mean=statistics.fmean(alignment),
                domain_high=100,
                method=method,
                formatter=lambda value: f"{value:.1f}%",
            )
        )
        body.extend(
            _matrix_cell(
                payload=payload,
                x=columns[1],
                y=y,
                values=utility,
                mean=statistics.fmean(utility),
                domain_high=100,
                method=method,
                formatter=lambda value: f"{value:.1f}%",
            )
        )
        query_values = [arm["metrics"]["teacher_query_ratio"] for arm in arms]
        if all(value is None for value in query_values):
            body.append(
                f'<text class="small" data-role="chart-label" data-applicable="false" x="{columns[2]}" y="{y + 6}">—  not applicable</text>'
            )
        elif any(value is None for value in query_values):
            raise ValueError(f"method {method} mixes applicable and non-applicable query ratios")
        else:
            query_percent = [100 * float(value) for value in query_values]
            body.extend(
                _matrix_cell(
                    payload=payload,
                    x=columns[2],
                    y=y,
                    values=query_percent,
                    mean=statistics.fmean(query_percent),
                    domain_high=100,
                    method=method,
                    formatter=lambda value: f"{value:.1f}%",
                )
            )
        body.extend(
            _matrix_cell(
                payload=payload,
                x=columns[3],
                y=y,
                values=time_values,
                mean=float(row["gpu_seconds_mean"]),
                domain_high=100,
                method=method,
                formatter=lambda value: f"{value:.1f}s",
            )
        )
        body.extend(
            _matrix_cell(
                payload=payload,
                x=columns[4],
                y=y,
                values=vram_values,
                mean=statistics.fmean(vram_values),
                main=float(row["peak_vram_bytes_max"]) / 2**30,
                domain_high=2,
                method=method,
                formatter=lambda value: f"{value:.2f} GiB",
            )
        )
    body.extend(
        [
            '<text class="small" x="48" y="684">Bars show the three-seed mean except VRAM, whose main bar is the observed maximum; seed shapes show every run.</text>',
            '<text class="small" x="48" y="707">Query ratio is selected target positions, not teacher FLOPs. DPO time includes its pinned TRL job.</text>',
        ]
    )
    return _svg_shell(
        "Outcome and continuation-cost matrix",
        (
            "Row matrix of alignment, retained tool utility, teacher-query ratio, continuation "
            "GPU time and peak VRAM for every method and seed."
        ),
        "Direct labels + seed marks · non-teacher methods remain not applicable, never zero",
        body,
        height=740,
    )


def _metric_coverage_matrix(payload: dict[str, Any]) -> str:
    summary = {row["method"]: row for row in payload["method_summary"]}
    baseline = {int(arm["seed"]): arm for arm in _method_arms(payload, "sft_checkpoint")}
    body: list[str] = [
        '<text class="header" x="48" y="144">Method</text>',
        '<text class="header" x="300" y="132">Harmful compliance</text>',
        '<text class="small" x="300" y="154">seed values</text>',
        '<text class="header" x="510" y="132">Over-refusal</text>',
        '<text class="small" x="510" y="154">seed values</text>',
        '<text class="header" x="665" y="132">Tool utility Δ</text>',
        '<text class="small" x="665" y="154">mean · seeds (pp)</text>',
        '<text class="header" x="850" y="132">Sandbox endpoint</text>',
        '<text class="small" x="850" y="154">measured?</text>',
        '<text class="header" x="1080" y="132" text-anchor="end">External safety</text>',
        '<text class="small" x="1080" y="154" text-anchor="end">executed?</text>',
    ]
    for index, method in enumerate(METHODS):
        y = 196 + index * 67
        arms = _method_arms(payload, method)
        harmful = [100 * float(arm["metrics"]["harmful_compliance_rate"]) for arm in arms]
        refusal = [100 * float(arm["metrics"]["over_refusal_rate"]) for arm in arms]
        utility_delta = [
            100
            * (
                float(arm["metrics"]["tool_utility_retention"])
                - float(baseline[int(arm["seed"])]["metrics"]["tool_utility_retention"])
            )
            for arm in arms
        ]
        body.extend(
            [
                f'<rect x="36" y="{y - 27}" width="1048" height="54" rx="9" fill="{("#0d192d" if index % 2 == 0 else "#0a1426")}"/>',
                f'<rect x="48" y="{y - 9}" width="12" height="18" rx="3" fill="{COLORS[method]}"/>',
                f'<text class="label" data-role="chart-label" x="70" y="{y + 6}">{escape(LABELS[method])}</text>',
                f'<text class="small" data-role="chart-label" data-encoding="seed-point" x="300" y="{y + 6}">{" · ".join(f"{v:.0f}%" for v in harmful)}</text>',
                f'<text class="small" data-role="chart-label" data-encoding="seed-point" x="510" y="{y + 6}">{" · ".join(f"{v:.0f}%" for v in refusal)}</text>',
                f'<text class="value" data-role="chart-label" x="665" y="{y + 6}">{_pp(100 * (float(summary[method]["tool_utility_retention_mean"]) - 1.0))}</text>',
                f'<text class="small" data-role="chart-label" data-encoding="seed-point" x="716" y="{y + 6}">{" / ".join(_pp(v) for v in utility_delta)}</text>',
                f'<text class="value" data-role="chart-label" x="850" y="{y + 6}" fill="#009E73">YES</text>',
                f'<text class="value" data-role="chart-label" x="1005" y="{y + 6}" fill="#E69F00">NOT RUN</text>',
            ]
        )
    body.extend(
        [
            '<rect x="48" y="594" width="1024" height="78" rx="14" fill="#111f37" stroke="#365276"/>',
            '<text class="value" data-role="chart-label" x="560" y="626" text-anchor="middle">The two sandbox safety checks tied at zero while utility still regressed.</text>',
            '<text class="small" data-role="chart-label" x="560" y="653" text-anchor="middle">IFEval, XSTest, HarmBench and RewardBench were not executed; this is not a broad safety benchmark.</text>',
        ]
    )
    return _svg_shell(
        "Metric coverage: zero sandbox failures did not imply full utility retention",
        (
            "Coverage matrix showing zero harmful-compliance and over-refusal rates, observed "
            "tool-utility deltas, measured sandbox endpoints and unexecuted external benchmarks."
        ),
        "All three measured seeds are printed · deterministic Minipolicy checks only",
        body,
    )


def assert_chart_suitability(figures: dict[str, str]) -> None:
    """Reject known misleading fallbacks in generated Alignment Lab figures."""
    expected = {
        "delta-from-sft.svg",
        "outcome-cost-matrix.svg",
        "metric-coverage-matrix.svg",
    }
    if set(figures) != expected:
        raise ValueError(f"Alignment Lab must publish exactly {sorted(expected)}")
    combined = "\n".join(figures.values()).lower()
    if "concentric" in combined or 'data-encoding="jitter"' in combined:
        raise ValueError("method-order rings and unlabelled jitter are forbidden")
    outcome = figures["outcome-cost-matrix.svg"]
    if "—  not applicable" not in outcome or 'data-applicable="false"' not in outcome:
        raise ValueError("non-teacher query ratios must remain explicitly not applicable")
    if 'data-applicable="false" data-value="0"' in outcome:
        raise ValueError("not-applicable query ratios must never be coerced to zero")
    if 'data-encoding="seed-point"' not in combined:
        raise ValueError("every chart set must expose the measured seed values")


def render_figures(payload: dict[str, Any], source_sha256: str) -> dict[str, str]:
    del source_sha256  # hashes live in the report provenance block, never in the plot canvas
    rendered = {
        "delta-from-sft.svg": _delta_from_sft(payload),
        "outcome-cost-matrix.svg": _outcome_cost_matrix(payload),
        "metric-coverage-matrix.svg": _metric_coverage_matrix(payload),
    }
    assert_chart_suitability(rendered)
    return rendered


def render_cards(payload: dict[str, Any], source_sha256: str) -> dict[str, str]:
    """Render one portable, corrected card per formal arm from the frozen result."""
    output: dict[str, str] = {}
    for arm in payload["arms"]:
        method = arm["method"]
        seed = arm["seed"]
        dpo_identity = arm["provenance"].get("dpo_identity")
        card = {
            "schema_version": 1,
            "benchmark": "alignment-lab-v1",
            "method": method,
            "seed": seed,
            "starting_sft_checkpoint": {
                "id": "mini-verl-qwen3-0.6b-tool-policy-sft",
                "sha256": payload["starting_checkpoint_sha256"],
            },
            "teacher": (
                None
                if method in {"sft_checkpoint", "continued_sft", "dpo"}
                else {
                    "id": "Qwen/Qwen3-0.6B",
                    "mode": "policy_conditioned",
                    "base_revision": payload["scope"]["model_revision"],
                    "adapter_revision": "7b98164f73e493c51f2ed3fca3169fea078f47f0",
                }
            ),
            "reference": (
                {
                    "kind": "implicit_initial_policy",
                    "trl_version": dpo_identity["trl_version"],
                    "base_model": dpo_identity["base_model"],
                    "adapter": dpo_identity["reference"],
                }
                if dpo_identity is not None
                else None
            ),
            "dpo_training": dpo_identity,
            "policy": {
                "id": "miniverl-tool-policy",
                "revision": "v1",
                "sha256": payload["policy_sha256"],
            },
            "metrics": arm["metrics"],
            "cost": arm["cost"],
            "limitations": [
                "One model family, one deterministic sandbox policy suite and one measured GPU.",
                "The common SFT checkpoint already saturated every measured policy and utility endpoint.",
                "Three seeds describe observed variation and do not establish a population claim.",
                "External IFEval, XSTest, HarmBench and RewardBench endpoints were not measured in this artifact.",
            ],
            "source_result_sha256": source_sha256,
            "raw_artifact_hashes": arm["artifacts"],
        }
        card["card_sha256"] = hashlib.sha256(canonical_json(card).encode("utf-8")).hexdigest()
        stem = f"{method}-seed-{seed}"
        output[f"{stem}.json"] = json.dumps(card, indent=2, sort_keys=True) + "\n"
        output[f"{stem}.md"] = (
            "# Alignment Card\n\n"
            f"Method: `{method}`\n\n"
            f"Seed: `{seed}`\n\n"
            f"Starting SFT checkpoint: `{payload['starting_checkpoint_sha256']}`\n\n"
            f"Policy: `miniverl-tool-policy@v1`\n\n"
            "## Measured endpoints\n\n"
            "```json\n" + json.dumps(arm["metrics"], indent=2, sort_keys=True) + "\n```\n\n"
            "## Cost\n\n"
            "```json\n" + json.dumps(arm["cost"], indent=2, sort_keys=True) + "\n```\n\n"
            "DPO cost includes the pinned external TRL training job when applicable. "
            "Evaluation time is not included in `gpu_seconds`.\n\n"
            "## Limitations\n\n" + "\n".join(f"- {item}" for item in card["limitations"]) + "\n\n"
            f"Source result SHA-256: `{source_sha256}`\n\n"
            f"Card content SHA-256: `{card['card_sha256']}`\n"
        )
    return output


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def render_report(payload: dict[str, Any], source_sha256: str) -> str:
    """Render the data-bound technical report in reviewable Markdown."""
    rows = []
    for item in payload["method_summary"]:
        rows.append(
            "| {label} | {alignment} | {harmful} | {over} | {utility} | {query} | "
            "{seconds:.1f} s | {vram:.2f} GiB |".format(
                label=item["label"],
                alignment=_percent(item["alignment_score_mean"]),
                harmful=_percent(item["harmful_compliance_rate_mean"]),
                over=_percent(item["over_refusal_rate_mean"]),
                utility=_percent(item["tool_utility_retention_mean"]),
                query=_percent(item["teacher_query_ratio_mean"]),
                seconds=float(item["gpu_seconds_mean"]),
                vram=float(item["peak_vram_bytes_max"]) / 2**30,
            )
        )
    diagnostic = payload["state_supervision_diagnostic"]["matched_comparisons"]
    frozen_hard = diagnostic["frozen_hard_vs_fresh_hard"]
    frozen_soft = diagnostic["frozen_soft_vs_fresh_soft"]
    fresh_hard_soft = diagnostic["fresh_hard_vs_fresh_soft"]
    baseline = next(row for row in payload["method_summary"] if row["method"] == "sft_checkpoint")
    regressions = [
        row["label"]
        for row in payload["method_summary"]
        if float(row["alignment_score_mean"]) < float(baseline["alignment_score_mean"])
    ]
    regression_sentence = (
        "Observed mean regressions were retained for: " + ", ".join(regressions) + "."
        if regressions
        else "No method regressed on the measured mean, but none could improve on the ceiling."
    )
    negative_arms = [
        arm
        for arm in payload["arms"]
        if float(arm["metrics"]["alignment_score"]) < float(baseline["alignment_score_mean"])
    ]
    if negative_arms:
        negative_rows = "\n".join(
            "- `{method}` seed `{seed}`: {score}; {failed} failed task(s), policy categories "
            "`{categories}`.".format(
                method=arm["method"],
                seed=arm["seed"],
                score=_percent(arm["metrics"]["alignment_score"]),
                failed=arm["provenance"]["final_failure_audit"]["failed_tasks"],
                categories=", ".join(arm["provenance"]["final_failure_audit"]["policy_categories"]),
            )
            for arm in negative_arms
        )
    else:
        negative_rows = "- None; all continuation arms tied the saturated baseline."
    return f"""# Alignment Lab v1: a saturated tool-policy case study

## Abstract

Alignment Lab v1 compares a frozen SFT checkpoint, continued alignment SFT,
DPO, offline soft teacher distillation, standard OPD and verifier-gated OPD
from one checksummed Qwen3-0.6B SFT checkpoint. The final test uses 48 paired
deterministic sandbox policy tasks and three preregistered seeds. Every method
uses the same starting policy and ordered final tasks. The SFT checkpoint
scored 100% policy compliance and 100% retained tool utility in every seed,
leaving no headroom for a continuation method to improve. No method beats that
baseline. {regression_sentence} The experiment therefore identifies cost,
query differences and method-specific regressions but no incremental
alignment-quality advantage. The correct pilot decision is to turn online
teacher querying off for this recipe, not to infer that OPD is broadly
ineffective.

## Study design

- Model: `Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca`
- Shared starting checkpoint: `{payload["starting_checkpoint_sha256"]}`
- Policy: Minipolicy v1, `{payload["policy_sha256"]}`
- Final test: 48 paired tasks per arm, seeds 1234, 20260727 and 20260801
- Budget: four continuation updates over four effective examples, except the
  frozen SFT checkpoint; DPO uses pinned TRL 1.8.0
- Hardware: one NVIDIA GeForce RTX 4080; cross-GPU generalization was not tested

The policy-conditioned teacher sees the private deterministic rubric. The
student never receives that context. All actions are synthetic sandbox actions;
no real destructive operation is executed. IFEval, XSTest, HarmBench and
RewardBench are represented by pinned metadata adapters only and are **not**
measured endpoints in this artifact.
“Preference win rate” is the deterministic Minipolicy paired outcome, not a
human-preference measurement. Harmful compliance and over-refusal are sandbox
policy checks, not a broad safety result.

## Final result

| method | alignment | harmful compliance | over-refusal | tool utility | teacher query ratio | continuation GPU time | peak VRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

![Forest chart of alignment and tool-utility deltas from the saturated SFT checkpoint, with every seed and the three-seed means](delta-from-sft.svg)

![Row matrix of alignment, retained tool utility, teacher-query ratio, continuation GPU time and peak VRAM, including every measured seed](outcome-cost-matrix.svg)

![Coverage matrix showing tied zero sandbox safety checks, utility regressions and external safety benchmarks not executed](metric-coverage-matrix.svg)

<details>
<summary>Figure provenance</summary>

- Result SHA-256: `{source_sha256}`
- Task-level result SHA-256: `{payload["task_results_sha256"]}`
- Three seed identities: `1234`, `20260727`, `20260801`

</details>

The starting checkpoint defines a ceiling; overlapping continuation points are
not evidence of algorithmic equivalence, and every non-overlapping regression
is retained. DPO cost includes its external pinned TRL training job; evaluation
time is excluded from the continuation-GPU-time axis. Teacher-query ratio
counts selected target positions and does not imply a proportional reduction
in teacher backbone FLOPs.

## State x Supervision diagnostic

The six-arm result directly observes oracle hard targets (continued SFT),
frozen-state preference supervision (DPO), frozen-state soft distributions
(offline distillation) and fresh-state soft distributions (standard OPD).
The frozen-soft and fresh-soft means are reported, but a task ceiling can make
their difference uninformative. The required hard-state comparisons remain
explicit:

- frozen hard vs fresh hard: teacher-argmax/student-token agreement
  `{float(frozen_hard["frozen"]):.4f}` vs `{float(frozen_hard["fresh"]):.4f}`
  (fresh minus frozen `{float(frozen_hard["fresh_minus_frozen"]):+.4f}`)
- frozen soft vs fresh soft: bucketed teacher entropy
  `{float(frozen_soft["frozen"]):.4f}` vs `{float(frozen_soft["fresh"]):.4f}` nats
  (fresh minus frozen `{float(frozen_soft["fresh_minus_frozen"]):+.4f}`)
- fresh hard vs fresh soft: matched soft targets retain
  `{100 * float(fresh_hard_soft["soft_probability_mass_beyond_argmax_mean"]):.3f}%`
  mean probability mass beyond the teacher argmax

No soft-target advantage is claimed. Verifier-gated OPD is separately treated
as a localized soft-supervision method, not relabeled as a hard-target cell.

## Verifier-gated OPD and pilot decision

The `policy-critical-span-v1` gate was calibrated on eval and frozen before the
test read. It records a decision for every example/span. Gating reduces queried
positions relative to standard OPD. Any resulting policy or utility regression
is retained rather than hidden; gating does not establish a general quality
gain or a proportional compute saving.

The versioned `alignment-pilot-v1` rule returns
`recommendation: insufficient_evidence`, followed by the operational decision:
do not spend online teacher-query cost on this already-saturated recipe. A more
discriminating policy suite would be required before choosing DPO, offline
distillation or either OPD variant.

The pilot binds 48 tasks, three seeds, measured continuation time, peak VRAM and
teacher-query fraction. Free-running teacher policy competence, the resulting
teacher-student policy gap, distribution-level top-k overlap, independent gate
precision and a population uncertainty interval were **not measured**; the JSON
records them as `null`, never as zero. The no-continuation result follows the
versioned less-than-2% headroom rule before those missing fields are consulted.

## Preserved deviation and negative evidence

Completed final-test regressions:

{negative_rows}

The first seed-1234 SFT baseline evaluated the first 24 tasks because the base
recipe initially allocated only 24 test tasks. The run then stopped before any
other method evaluated test. Preregistration revision 1.4 publicly froze a
recovery rule: preserve tasks 0-23, evaluate only tasks 24-47 once, and combine
the disjoint segments. The result contains all 48 unique task IDs with zero
repeated test tasks. The interrupted continued-SFT construction run is retained
outside the headline result and was never evaluated.

The primary negative finding is preserved: no continuation method improves the
already-saturated SFT checkpoint, and any completed regression remains in the
headline result. It would be misleading to turn a lower teacher-query ratio
into a quality claim.

## Scope and limitations

- One small model family, one deterministic sandbox policy suite, one GPU and
  three seeds do not support broad safety or population claims.
- The suite's deterministic validators are valuable for auditability but are
  too easy for the common SFT checkpoint.
- External safety, preference, instruction-following and general-capability
  suites were not executed; their licenses/revisions are metadata only here.
- GPU time and peak VRAM are observed values for this machine and software
  stack, not forecasts for other cards.
- Localized or verifier-qualified distillation is not claimed as novel.

## Reproducibility and artifacts

- Result SHA-256: `{source_sha256}`
- Task-level result SHA-256: `{payload["task_results_sha256"]}`
- Preregistration SHA-256: `{payload["preregistration_sha256"]}`
- Immutable calculator artifact SHA-256: `{payload["frozen_calculator_sha256"]}`

Regenerate and compare every figure and public Alignment Card with:

```bash
python scripts/publish_alignment_lab_artifacts.py --check
```

The machine-readable result records raw run-artifact hashes, DPO provenance,
the two-segment recovery, all 18 completed arms and all 864 task-level rows.
"""


def _check(result_path: Path, task_path: Path) -> None:
    payload = _load_result(result_path)
    if payload.get("task_results_sha256") != _sha256(task_path):
        raise ValueError("Alignment Lab task-results digest mismatch")
    rendered = render_figures(payload, _sha256(result_path))
    for name, content in rendered.items():
        target = DOCS / name
        if not target.is_file() or target.read_text(encoding="utf-8") != content:
            raise ValueError(f"generated artifact is stale: {target}")
    for name, content in render_cards(payload, _sha256(result_path)).items():
        target = CARDS / name
        if not target.is_file() or target.read_text(encoding="utf-8") != content:
            raise ValueError(f"generated artifact is stale: {target}")
    report = render_report(payload, _sha256(result_path))
    target = DOCS / "alignment-lab-v1.md"
    if not target.is_file() or target.read_text(encoding="utf-8") != report:
        raise ValueError(f"generated artifact is stale: {target}")
    pilot = json.dumps(payload["pilot"], indent=2, sort_keys=True) + "\n"
    if not PILOT_EXAMPLE.is_file() or PILOT_EXAMPLE.read_text(encoding="utf-8") != pilot:
        raise ValueError(f"generated artifact is stale: {PILOT_EXAMPLE}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--baseline-prefix", type=Path)
    parser.add_argument("--baseline-recovery", type=Path)
    parser.add_argument("--dpo-root", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--result", type=Path, default=RESULT)
    parser.add_argument("--task-results", type=Path, default=TASK_RESULTS)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        _check(args.result, args.task_results)
        return 0
    missing = [
        name
        for name in ("run_root", "baseline_prefix", "baseline_recovery")
        if getattr(args, name) is None
    ]
    if missing:
        parser.error(
            "publication requires " + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        )
    hashes = publish(
        run_root=args.run_root,
        baseline_prefix=args.baseline_prefix,
        baseline_recovery=args.baseline_recovery,
        dpo_root=args.dpo_root,
        result_path=args.result,
        task_path=args.task_results,
    )
    print(json.dumps(hashes, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
