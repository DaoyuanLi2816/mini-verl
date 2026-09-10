"""Generate the direct-config audit from immutable upstream corpus bytes."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections import Counter
from pathlib import Path

import yaml

from miniverl.bridge.direct import DIRECT_PROFILE, _put, compile_direct, direct_rules_digest

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "benchmarks/compatibility/verl-v0.9-direct-v4/manifest.json"
VARIANTS = {
    "ppo-sequence-sum": ("ppo", {"actor_rollout_ref.actor.loss_agg_mode": "seq-mean-token-sum"}),
    "grpo-sampled-validation": (
        "grpo",
        {
            "actor_rollout_ref.rollout.val_kwargs.do_sample": True,
            "actor_rollout_ref.rollout.val_kwargs.temperature": 0.8,
            "actor_rollout_ref.rollout.val_kwargs.n": 2,
        },
    ),
    "grpo-group-selection": ("grpo", {"algorithm.filter_groups.enable": True}),
    "grpo-oversampling": ("grpo", {"actor_rollout_ref.rollout.over_sample_rate": 0.5}),
    "ppo-shuffled-minibatches": ("ppo", {"actor_rollout_ref.actor.shuffle": True}),
}


def generated() -> bytes:
    cases = []
    originals = {
        path.name.removesuffix("-complete.yaml"): path
        for path in sorted((ROOT / "benchmarks/compatibility/verl-v0.9.0").glob("*-complete.yaml"))
    }
    with tempfile.TemporaryDirectory(prefix="miniverl-direct-corpus-") as directory:
        inputs = [(name, path, None) for name, path in originals.items()]
        for name, (base, changes) in VARIANTS.items():
            source = yaml.safe_load(originals[base].read_text(encoding="utf-8"))
            for key, value in changes.items():
                _put(source, key, value)
            path = Path(directory) / f"{name}.yaml"
            path.write_text(yaml.safe_dump(source, sort_keys=True), encoding="utf-8", newline="\n")
            inputs.append((name, path, {"source_case": base, "overrides": changes}))
        for name, path, adaptation in inputs:
            report = compile_direct(path)
            decisions = {row["field"]: row["classification"] for row in report["fields"]}
            cases.append(
                {
                    "case": name,
                    "kind": "derived_stress_config" if adaptation else "complete_official_config",
                    "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "adaptation": adaptation,
                    "semantic_status": report["semantic_status"],
                    "execution_status": "not_executed"
                    if report["semantic_status"] == "accepted"
                    else "unsupported_semantics",
                    "external_bindings": [
                        "data.train_files",
                        "data.val_files",
                        *([] if name == "trained-rm-grpo" else ["reward.function"]),
                    ],
                    "exact_model_hardware": "not_assessed",
                    "field_count": report["source_leaf_count"],
                    "field_classification": decisions,
                    "classification_counts": dict(sorted(Counter(decisions.values()).items())),
                    "rejections": [
                        {"field": row["field"], "reason": row["reason"]}
                        for row in report["rejections"]
                    ],
                    "native_ir_schema_validated": report["ir_schema_validation_passed"],
                }
            )
    payload = {
        "schema_version": 1,
        "profile": DIRECT_PROFILE,
        "field_rules_sha256": direct_rules_digest(),
        "upstream_tag": "v0.9.0",
        "upstream_commit": report["upstream_commit"],
        "scope": "static semantic compilation; neither exact-model execution nor a scientific benchmark",
        "case_count": len(cases),
        "semantic_accepted": sum(case["semantic_status"] == "accepted" for case in cases),
        "direct_execution_count": 0,
        "hardware_blocked_count": None,
        "qualification": "separately exact-wheel qualified bounded PPO/GRPO examples with explicit reward bindings",
        "cases": cases,
    }
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    content = generated()
    if args.check:
        if not DESTINATION.is_file() or DESTINATION.read_bytes() != content:
            raise SystemExit(
                "direct compatibility audit is stale; run scripts/publish_direct_compatibility.py"
            )
    else:
        DESTINATION.parent.mkdir(parents=True, exist_ok=True)
        DESTINATION.write_bytes(content)


if __name__ == "__main__":
    main()
