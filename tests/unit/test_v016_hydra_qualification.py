"""Release promotion requires native composition and exact replay evidence."""

import copy
from functools import lru_cache
from types import SimpleNamespace as NS

import pytest

from miniverl.bridge.direct_v5 import PROFILE, rules_digest
from miniverl.bridge.hydra import compose_verl
from miniverl.bridge.hydra_examples import qualification_overrides
from miniverl.qualification_hydra import validate_hydra_evidence


@lru_cache
def fixture():
    qualification = NS(
        source_commit="a" * 40,
        miniverl_version="0.16.0",
        wheel=NS(sha256="b" * 64),
        environment=NS(gpu_name="RTX 4080", packages={"torch": "2.13.0"}),
    )
    payload = {
        "schema_version": 1,
        "kind": "installed_native_hydra_workflows",
        "status": "passed",
        "source_commit": qualification.source_commit,
        "miniverl_version": qualification.miniverl_version,
        "wheel_sha256": qualification.wheel.sha256,
        "installed_wheel_verified": True,
        "source_dirty": False,
        "rehearsal": False,
        "environment": {"gpu_name": "RTX 4080", "torch": "2.13.0"},
        "cases": {},
        "commands": [{"command": ["miniverl", "data"], "exit_code": 0}],
    }
    for name in ("ppo", "grpo"):
        payload["cases"][name] = {
            "profile": PROFILE,
            "field_rules_sha256": rules_digest(),
            "composition": compose_verl(
                config_name="ppo_trainer", overrides=qualification_overrides(name)
            ).provenance,
            "composition_status": "resolved",
            "semantic_status": "accepted",
            "execution_status": "completed",
            "resolved_bytes_preserved": True,
            "manual_native_yaml_edits": False,
            "exact_tensor_resume": True,
            "exact_semantic_journal_resume": True,
            "journal_normalization": "rewards.jsonl: omit top-level duration_ms only; other journals byte-exact",
            "handoff_verdict": "ok",
            "launchable": False,
            "distributed_execution_tested": False,
            "updates": {"actor_records": 4, "critic_records": 4 if name == "ppo" else 0},
            "sampling_semantics": {
                "actor_shuffle": True,
                "actor_seed": 42,
                "critic_shuffle": True,
                "critic_seed": 113,
                "filter_metric": "score" if name == "grpo" else None,
                "refill_failed_groups": True,
                "max_attempted_groups": 64,
            },
            "runtime_bindings": [
                {"binding": "reward.provider", "bound": "target_length"},
                {"binding": "sampling.max_attempted_groups", "bound": 64},
            ],
            "hardware_plan": {"status": "executable_with_local_lowering"},
            "sampling": [
                {
                    "retained_groups": 2,
                    "attempted_groups": 2,
                    "selected_trajectory_ids": [str(i) for i in range(c * 4, c * 4 + 4)],
                }
                for c in range(2)
            ],
            "minibatches": [
                {
                    "ppo_epoch": e,
                    "minibatch_trajectory_ids": [str(i) for i in range(c * 4, c * 4 + 4)],
                }
                for c in range(2)
                for e in range(2)
                for _ in range(2 if name == "ppo" else 1)
            ],
            "memory": {"status": "measured", "peak_reserved_gib": 2.0},
            "tensor_hashes": dict.fromkeys(
                [
                    "adapter.safetensors",
                    "optimizer.safetensors",
                    "critic.safetensors",
                    "critic-optimizer.safetensors",
                ],
                "c" * 64,
            ),
            "journal_hashes": dict.fromkeys(
                ["trajectories.jsonl", "rewards.jsonl", "advantages.jsonl"], "d" * 64
            ),
            "ir_sha256": "e" * 64,
        }
        for index, command in enumerate(
            [
                "run",
                "run",
                "inspect",
                "report",
                "run",
                "inspect",
                "export-adapter",
                "export-verl",
                "bridge",
            ]
        ):
            words = ["miniverl", command]
            if command == "run":
                options = (
                    ["--dry-run"]
                    if index == 0
                    else ["--resume-from", "checkpoint"]
                    if index == 4
                    else []
                )
                words += [
                    "--verl-config-name",
                    "ppo_trainer",
                    "--bind",
                    "reward.provider=target_length",
                    "--bind",
                    "sampling.max_attempted_groups=64",
                    *options,
                    *qualification_overrides(name),
                    "--json",
                ]
            elif command == "bridge":
                words += ["doctor", "bundle"]
            payload["commands"].append({"command": words, "exit_code": 0})
    return payload, qualification


def test_complete_native_hydra_evidence():
    validate_hydra_evidence(*fixture())


@pytest.mark.parametrize(
    "key,value",
    [
        ("composition_status", "failed"),
        ("composition", {}),
        ("field_rules_sha256", "a" * 64),
        ("exact_tensor_resume", False),
        ("sampling", []),
        ("minibatches", []),
        ("sampling_semantics", {}),
        ("tensor_hashes", {}),
        ("journal_hashes", {}),
        ("journal_normalization", "ignore all rewards"),
        ("launchable", True),
        ("memory", {"status": "measured", "peak_reserved_gib": float("nan")}),
    ],
)
def test_incomplete_or_misbound_native_evidence_rejects(key, value):
    payload, qualification = copy.deepcopy(fixture())
    payload["cases"]["grpo"][key] = value
    with pytest.raises(ValueError):
        validate_hydra_evidence(payload, qualification)


@pytest.mark.parametrize("mutation", ["wheel", "dirty", "native_train", "overrides"])
def test_candidate_identity_and_native_transcript(mutation):
    payload, qualification = copy.deepcopy(fixture())
    if mutation == "wheel":
        payload["wheel_sha256"] = "f" * 64
    elif mutation == "dirty":
        payload["source_dirty"] = True
    elif mutation == "native_train":
        payload["commands"][2]["command"][1] = "train"
    else:
        payload["commands"][2]["command"][-2] = "algorithm.adv_estimator=rloo"
    with pytest.raises(ValueError):
        validate_hydra_evidence(payload, qualification)
