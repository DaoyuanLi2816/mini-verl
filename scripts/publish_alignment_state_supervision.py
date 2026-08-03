#!/usr/bin/env python3
"""Publish the matched State x Supervision signal diagnostic for Alignment Lab v1."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

from miniverl.cache.store import TeacherCache
from miniverl.utils.runs import canonical_json

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "benchmarks/results/alignment-lab-v1-state-supervision.json"
PREREGISTRATION = ROOT / "benchmarks/preregistration/alignment-lab-v1.yaml"
PREREGISTRATION_SHA256 = "71307dbfe9a5bb20c686307cafce8bd254c07af8b69c1bf1c6ec0dbf53a8cde0"
STARTING_CHECKPOINT_SHA256 = "7304922281268a687dd1c75ba918e26c64c8207b5701db78c368afd20d80ae89"
SEEDS = (1234, 20260727, 20260801)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _cache_summary(path: Path) -> dict[str, Any]:
    cache = TeacherCache.open(path)
    matches = 0
    positions = 0
    top1_values: list[float] = []
    tail_values: list[float] = []
    entropy_values: list[float] = []
    weights_total = 0.0
    for trajectory_id in sorted(cache.index.entries):
        batch = cache.read(trajectory_id)
        active = batch.weights > 0
        count = int(active.sum().item())
        positions += count
        weights_total += float(batch.weights[active].sum().item())
        matches += int(
            (batch.topk_indices[active, 0] == batch.target_token_ids[active]).sum().item()
        )
        topk_probability = batch.topk_log_probs[active].exp()
        tail_probability = batch.tail_log_prob[active].exp()
        top1_values.extend(float(value) for value in topk_probability[:, 0].tolist())
        tail_values.extend(float(value) for value in tail_probability.tolist())
        entropy = -(topk_probability * batch.topk_log_probs[active]).sum(dim=-1)
        entropy -= (tail_probability * batch.tail_log_prob[active]).nan_to_num(
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        entropy_values.extend(float(value) for value in entropy.tolist())
    if positions == 0:
        raise ValueError(f"diagnostic cache has no selected positions: {path}")
    entry_binding = [
        {
            "trajectory_id": trajectory_id,
            "policy_version": entry.policy_version,
            "checksum": entry.checksum,
            "positions": entry.num_positions,
        }
        for trajectory_id, entry in sorted(cache.index.entries.items())
    ]
    teacher = {
        "model_id": cache.index.teacher_model_id,
        "model_revision": cache.index.teacher_model_revision,
        "adapter": cache.index.teacher_adapter_provenance,
        "temperature": cache.index.temperature,
        "top_k": cache.index.top_k,
        "loss_mode": cache.index.loss_mode,
    }
    return {
        "trajectories": len(cache),
        "selected_positions": positions,
        "weights_total": weights_total,
        "policy_versions": sorted(cache.index.policy_versions()),
        "state_digest": _digest(entry_binding),
        "teacher_digest": _digest(teacher),
        "teacher_argmax_matches_student_token_rate": matches / positions,
        "teacher_top1_probability_mean": statistics.fmean(top1_values),
        "teacher_off_argmax_probability_mean": statistics.fmean(
            1.0 - value for value in top1_values
        ),
        "teacher_tail_probability_mean": statistics.fmean(tail_values),
        "bucketed_teacher_entropy_nats_mean": statistics.fmean(entropy_values),
        "cache_index_sha256": _sha256(path / "index.json"),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positions = sum(int(row["selected_positions"]) for row in rows)

    def weighted(field: str) -> float:
        return sum(float(row[field]) * int(row["selected_positions"]) for row in rows) / positions

    return {
        "seeds": list(SEEDS),
        "trajectories": sum(int(row["trajectories"]) for row in rows),
        "selected_positions": positions,
        "policy_versions": sorted({version for row in rows for version in row["policy_versions"]}),
        "state_digest": _digest([row["state_digest"] for row in rows]),
        "teacher_digest": _digest([row["teacher_digest"] for row in rows]),
        "teacher_argmax_matches_student_token_rate": weighted(
            "teacher_argmax_matches_student_token_rate"
        ),
        "teacher_top1_probability_mean": weighted("teacher_top1_probability_mean"),
        "teacher_off_argmax_probability_mean": weighted("teacher_off_argmax_probability_mean"),
        "teacher_tail_probability_mean": weighted("teacher_tail_probability_mean"),
        "bucketed_teacher_entropy_nats_mean": weighted("bucketed_teacher_entropy_nats_mean"),
        "per_seed": rows,
    }


def publish(*, run_root: Path, output: Path = OUTPUT) -> dict[str, Any]:
    if _sha256(PREREGISTRATION) != PREREGISTRATION_SHA256:
        raise ValueError("Alignment Lab preregistration digest changed")
    frozen_rows = [
        {
            "seed": seed,
            **_cache_summary(run_root / f"test-offline_distillation-seed-{seed}" / "teacher-cache"),
        }
        for seed in SEEDS
    ]
    fresh_rows = [
        {
            "seed": seed,
            **_cache_summary(run_root / f"test-standard_opd-seed-{seed}" / "teacher-cache"),
        }
        for seed in SEEDS
    ]
    frozen = _aggregate(frozen_rows)
    fresh = _aggregate(fresh_rows)
    budget_digest = _digest(
        {
            "optimizer_updates": 4,
            "effective_examples_per_update": 4,
            "student_seeds": list(SEEDS),
            "starting_checkpoint_sha256": STARTING_CHECKPOINT_SHA256,
        }
    )
    common = {
        "teacher_digest": frozen["teacher_digest"],
        "starting_checkpoint_digest": STARTING_CHECKPOINT_SHA256,
        "budget_digest": budget_digest,
        "seeds": list(SEEDS),
    }
    if fresh["teacher_digest"] != frozen["teacher_digest"]:
        raise ValueError("frozen and fresh diagnostics use different teachers")
    cells = []
    for state_source, state in (("frozen_student", frozen), ("fresh_student", fresh)):
        signal = {
            key: state[key]
            for key in (
                "trajectories",
                "selected_positions",
                "policy_versions",
                "teacher_argmax_matches_student_token_rate",
                "teacher_top1_probability_mean",
                "teacher_off_argmax_probability_mean",
                "teacher_tail_probability_mean",
                "bucketed_teacher_entropy_nats_mean",
            )
        }
        cells.extend(
            [
                {
                    "state_source": state_source,
                    "supervision": "teacher_argmax",
                    "state_digest": state["state_digest"],
                    **common,
                    "signal": {
                        **signal,
                        "target_categories_per_position": 1,
                        "retained_teacher_probability_mass_mean": state[
                            "teacher_top1_probability_mean"
                        ],
                    },
                },
                {
                    "state_source": state_source,
                    "supervision": "teacher_soft_distribution",
                    "state_digest": state["state_digest"],
                    **common,
                    "signal": {
                        **signal,
                        "target_categories_per_position": 65,
                        "retained_teacher_probability_mass_mean": 1.0,
                    },
                },
            ]
        )
    payload = {
        "schema_version": 1,
        "name": "alignment-lab-v1-state-supervision",
        "measurement_status": "measured_signal_diagnostic_not_training_outcome",
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "starting_checkpoint_sha256": STARTING_CHECKPOINT_SHA256,
        "hard_definition": "teacher_argmax",
        "source_scope": {
            "frozen_student": "all fixed starting-policy cache entries",
            "fresh_student": (
                "the two most recent policy versions retained by the preregistered cache policy"
            ),
            "selection_timing": "cache retention was configured before final test",
        },
        "cells": cells,
        "matched_comparisons": {
            "frozen_hard_vs_fresh_hard": {
                "endpoint": "teacher_argmax_matches_student_token_rate",
                "frozen": frozen["teacher_argmax_matches_student_token_rate"],
                "fresh": fresh["teacher_argmax_matches_student_token_rate"],
                "fresh_minus_frozen": fresh["teacher_argmax_matches_student_token_rate"]
                - frozen["teacher_argmax_matches_student_token_rate"],
                "interpretation": "state-source change in hard-target disagreement signal",
            },
            "frozen_soft_vs_fresh_soft": {
                "endpoint": "bucketed_teacher_entropy_nats_mean",
                "frozen": frozen["bucketed_teacher_entropy_nats_mean"],
                "fresh": fresh["bucketed_teacher_entropy_nats_mean"],
                "fresh_minus_frozen": fresh["bucketed_teacher_entropy_nats_mean"]
                - frozen["bucketed_teacher_entropy_nats_mean"],
                "interpretation": "state-source change in soft-distribution signal",
            },
            "fresh_hard_vs_fresh_soft": {
                "same_state_digest": True,
                "same_teacher_digest": True,
                "same_budget_digest": True,
                "same_starting_checkpoint_digest": True,
                "same_seeds": True,
                "soft_probability_mass_beyond_argmax_mean": fresh[
                    "teacher_off_argmax_probability_mean"
                ],
                "interpretation": (
                    "probability information present in the matched soft target and discarded "
                    "by its argmax projection"
                ),
            },
        },
        "headline_training_cells": {
            "oracle_hard_target": "continued_sft",
            "frozen_preference_reward": "dpo",
            "frozen_soft_distribution": "offline_distillation",
            "fresh_soft_distribution": "standard_opd",
            "localized_fresh_soft_distribution": "verifier_gated_opd",
        },
        "claims": {
            "state_or_soft_target_quality_advantage": "not_claimed",
            "reason": (
                "This artifact measures matched target signal, not separately trained hard-target "
                "policies; downstream quality comes only from the six-arm final benchmark."
            ),
        },
        "source_artifacts": {
            "frozen_cache_index_sha256_by_seed": {
                str(row["seed"]): row["cache_index_sha256"] for row in frozen_rows
            },
            "fresh_cache_index_sha256_by_seed": {
                str(row["seed"]): row["cache_index_sha256"] for row in fresh_rows
            },
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=ROOT / "runs/alignment-lab-v1-final-v2")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        current = args.output.read_text(encoding="utf-8")
        temporary = args.output.with_suffix(".check.json")
        try:
            publish(run_root=args.run_root, output=temporary)
            if temporary.read_text(encoding="utf-8") != current:
                raise SystemExit(f"generated artifact is stale: {args.output}")
        finally:
            temporary.unlink(missing_ok=True)
        return 0
    payload = publish(run_root=args.run_root, output=args.output)
    print(
        json.dumps(
            {str(args.output): _sha256(args.output), "cells": len(payload["cells"])}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
