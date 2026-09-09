"""Product publication requires candidate-bound complete command evidence."""

from types import SimpleNamespace as NS

import pytest

from miniverl.bridge.rl_v09 import VERL_RL_V09_PRODUCT_PROFILE, rl_v09_field_rules_digest
from miniverl.qualification_product import PRODUCT_COMMANDS, validate_product_evidence


def evidence():
    qualification = NS(
        source_commit="a" * 40,
        miniverl_version="0.14.0",
        wheel=NS(sha256="b" * 64),
        environment=NS(gpu_name="RTX 4080", packages={"torch": "2.13.0"}),
    )
    payload = {
        "schema_version": 1,
        "kind": "installed_ppo_grpo_product_workflows",
        "status": "passed",
        "source_commit": qualification.source_commit,
        "miniverl_version": qualification.miniverl_version,
        "wheel_sha256": qualification.wheel.sha256,
        "installed_wheel_verified": True,
        "source_dirty": False,
        "rehearsal": False,
        "environment": {"gpu_name": "RTX 4080", "torch": "2.13.0"},
        "commands": [{"command": ["miniverl", name], "exit_code": 0} for name in PRODUCT_COMMANDS],
        "cases": {},
    }
    for offset, name, count in ((1, "ppo", 4), (13, "grpo", 2)):
        payload["cases"][name] = {
            "profile": VERL_RL_V09_PRODUCT_PROFILE,
            "field_rules_sha256": rl_v09_field_rules_digest(VERL_RL_V09_PRODUCT_PROFILE),
            "exact_tensor_resume": True,
            "resume_counts_verified": True,
            "launchable": False,
            "distributed_execution_tested": False,
            "import_status": "accepted",
            "handoff_verdict": "ok",
            "updates": {"actor_records": count, "critic_records": count if name == "ppo" else 0},
            "rollouts": {"trajectories": 8},
            "rewards": {"count": 8},
            "memory": {"status": "measured", "peak_reserved_gib": 3.0},
            "tensor_hashes": dict.fromkeys(
                [
                    "adapter.safetensors",
                    "optimizer.safetensors",
                    "critic.safetensors",
                    "critic-optimizer.safetensors",
                ],
                "c" * 64,
            ),
        }
        payload["commands"][offset]["command"] += [
            "--profile",
            VERL_RL_V09_PRODUCT_PROFILE,
            "--example",
            name,
        ]
        payload["commands"][offset + 2]["command"] += ["--dry-run"]
        payload["commands"][offset + 7]["command"] += ["--resume-from"]
        payload["commands"][offset + 11]["command"] += ["doctor"]
    return payload, qualification


def test_complete_candidate_evidence_passes():
    validate_product_evidence(*evidence())


@pytest.mark.parametrize(
    "key,value",
    [
        ("rehearsal", True),
        ("source_dirty", True),
        ("installed_wheel_verified", False),
        ("wheel_sha256", "d" * 64),
        ("source_commit", "e" * 40),
    ],
)
def test_rejects_rehearsal_or_wrong_candidate(key, value):
    payload, qualification = evidence()
    payload[key] = value
    with pytest.raises(ValueError):
        validate_product_evidence(payload, qualification)


@pytest.mark.parametrize(
    "key,value",
    [
        ("exact_tensor_resume", False),
        ("launchable", True),
        ("tensor_hashes", {}),
        ("memory", {"status": "estimated", "peak_reserved_gib": 2}),
        ("updates", {"actor_records": 4, "critic_records": 0}),
        ("rollouts", {"trajectories": 7}),
    ],
)
def test_rejects_incomplete_or_misrepresented_ppo(key, value):
    payload, qualification = evidence()
    payload["cases"]["ppo"][key] = value
    with pytest.raises(ValueError):
        validate_product_evidence(payload, qualification)


def test_no_resume_transcript_is_not_complete():
    payload, qualification = evidence()
    payload["commands"][8]["command"] = ["miniverl", "train"]
    with pytest.raises(ValueError, match="resume"):
        validate_product_evidence(payload, qualification)
