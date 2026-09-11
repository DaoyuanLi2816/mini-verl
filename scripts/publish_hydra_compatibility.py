"""Compose the official launch corpus through the native, pinned adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections import Counter
from pathlib import Path

import yaml

from miniverl.bridge.direct_v5 import PROFILE, compile_direct, rules_digest
from miniverl.bridge.hydra import compose_verl
from miniverl.errors import ConfigError

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "benchmarks/compatibility/verl-v0.9-hydra-v5/manifest.json"
VARIANTS = {
    "ppo-shuffled-minibatches": (
        "ppo",
        [
            "actor_rollout_ref.actor.shuffle=true",
            "critic.shuffle=true",
            "critic.data_loader_seed=113",
        ],
    ),
    "grpo-group-selection": (
        "grpo",
        ["algorithm.filter_groups.enable=true", "algorithm.filter_groups.metric=score"],
    ),
    "grpo-refill": ("grpo", ["trainer.v1.sampler.sync_refill_failed_groups=true"]),
    "grpo-filter-refill": (
        "grpo",
        [
            "algorithm.filter_groups.enable=true",
            "algorithm.filter_groups.metric=score",
            "trainer.v1.sampler.sync_refill_failed_groups=true",
            "algorithm.filter_groups.max_inflight_gen_batches=2",
        ],
    ),
    "grpo-legacy-over-sample": ("grpo", ["actor_rollout_ref.rollout.over_sample_rate=0.5"]),
    "grpo-missing-filter-metric": ("grpo", ["algorithm.filter_groups.enable=true"]),
    "ppo-unsupported-entropy-estimator": (
        "ppo",
        ["actor_rollout_ref.actor.policy_loss.loss_mode=gpg"],
    ),
    "invalid-sweep": ("grpo", ["actor_rollout_ref.rollout.n=2,4"]),
    "invalid-interpolation": ("grpo", ["++local=${unknown_variable}"]),
}


def generated() -> bytes:
    originals = json.loads(
        (ROOT / "benchmarks/compatibility/verl-v0.9.0/manifest.json").read_text()
    )["cases"]
    launches = {case["name"]: case["launch_arguments"] for case in originals}
    inputs = [(name, name, []) for name in launches] + [
        (name, base, extra) for name, (base, extra) in VARIANTS.items()
    ]
    cases = []
    with tempfile.TemporaryDirectory(prefix="miniverl-hydra-corpus-") as directory:
        for name, base, extra in inputs:
            arguments = launches[base] + extra
            case = {
                "name": name,
                "source_case": base,
                "config_name": "ppo_trainer",
                "overrides": arguments,
                "hardware_status": "not_assessed",
                "executed": False,
            }
            try:
                composition = compose_verl(config_name="ppo_trainer", overrides=arguments)
            except ConfigError as exc:
                case.update(
                    composition_status="failed",
                    semantic_status="not_assessed",
                    execution_status="not_assessed",
                    reason=str(exc),
                )
                cases.append(case)
                continue
            source = Path(directory) / "resolved.yaml"
            source.write_bytes(composition.resolved_yaml)
            report = compile_direct(source)
            if not extra:
                expected = yaml.safe_load(
                    (ROOT / f"benchmarks/compatibility/verl-v0.9.0/{base}-complete.yaml").read_text(
                        encoding="utf-8"
                    )
                )
                if yaml.safe_load(composition.resolved_yaml) != expected:
                    raise ValueError(
                        f"native composition differs from trusted upstream corpus: {base}"
                    )
            case.update(
                composition_status="resolved",
                semantic_status=report["semantic_status"],
                execution_status=report["execution_status"],
                input_tree_sha256=composition.provenance["input_tree_sha256"],
                resolved_sha256=hashlib.sha256(composition.resolved_yaml).hexdigest(),
                trusted_composition_equal=True if not extra else None,
                required_bindings=report["required_bindings"],
                rejections=report["rejections"],
                classification_counts=dict(
                    sorted(Counter(row["classification"] for row in report["fields"]).items())
                ),
                sampling_semantics=report.get("sampling_semantics"),
            )
            cases.append(case)
    payload = {
        "schema_version": 1,
        "profile": PROFILE,
        "field_rules_sha256": rules_digest(),
        "upstream_tag": composition.provenance["upstream_tag"],
        "upstream_commit": composition.provenance["upstream_commit"],
        "case_count": len(cases),
        "composition_resolved": sum(c["composition_status"] == "resolved" for c in cases),
        "semantic_accepted": sum(c["semantic_status"] == "accepted" for c in cases),
        "execution_count": 0,
        "scope": "Native composition and semantic audit; exact-model hardware and execution are separate qualification axes.",
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
            raise SystemExit("native Hydra compatibility artifact is stale")
    else:
        DESTINATION.parent.mkdir(parents=True, exist_ok=True)
        DESTINATION.write_bytes(content)


if __name__ == "__main__":
    main()
