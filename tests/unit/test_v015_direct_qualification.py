"""Direct release promotion rejects incomplete and misbound candidate evidence."""

import hashlib
from importlib.resources import files
from types import SimpleNamespace as NS

import pytest

from miniverl.bridge.direct import DIRECT_PROFILE, direct_rules_digest
from miniverl.qualification_direct import DIRECT_COMMANDS, validate_direct_evidence


def evidence():
    qualification = NS(
        source_commit="a" * 40,
        miniverl_version="0.15.0",
        wheel=NS(sha256="b" * 64),
        environment=NS(gpu_name="RTX 4080", packages={"torch": "2.13.0"}),
    )
    payload = {
        "schema_version": 1,
        "kind": "installed_direct_verl_workflows",
        "status": "passed",
        "source_commit": qualification.source_commit,
        "miniverl_version": qualification.miniverl_version,
        "wheel_sha256": qualification.wheel.sha256,
        "installed_wheel_verified": True,
        "source_dirty": False,
        "rehearsal": False,
        "environment": {"gpu_name": "RTX 4080", "torch": "2.13.0"},
        "commands": [{"command": ["miniverl", name], "exit_code": 0} for name in DIRECT_COMMANDS],
        "cases": {},
    }
    for offset, name in ((1, "ppo"), (10, "grpo")):
        payload["cases"][name] = {
            "profile": DIRECT_PROFILE,
            "field_rules_sha256": direct_rules_digest(),
            "semantic_status": "accepted",
            "execution_status": "completed",
            "source_config_sha256": hashlib.sha256(
                files("miniverl.resources").joinpath(f"verl_direct_{name}.yaml").read_bytes()
            ).hexdigest(),
            "source_bytes_preserved": True,
            "manual_native_yaml_edits": False,
            "exact_tensor_resume": True,
            "resume_counts_verified": True,
            "handoff_verdict": "ok",
            "launchable": False,
            "distributed_execution_tested": False,
            "loss_aggregation": "token-mean" if name == "ppo" else "seq-mean-token-mean",
            "updates": {"actor_records": 4, "critic_records": 4 if name == "ppo" else 0},
            "rollouts": {"trajectories": 8},
            "rewards": {"count": 8},
            "schedule": {"cycles": 2},
            "hardware_plan": {"status": "executable_with_local_lowering"},
            "runtime_bindings": [{"binding": "reward.provider", "bound": "target_length"}],
            "memory": {"status": "measured", "peak_reserved_gib": 4.0},
            "ir_sha256": "c" * 64,
            "tensor_hashes": dict.fromkeys(
                [
                    "adapter.safetensors",
                    "optimizer.safetensors",
                    "critic.safetensors",
                    "critic-optimizer.safetensors",
                ],
                "d" * 64,
            ),
        }
        for index in (offset, offset + 1, offset + 4):
            payload["commands"][index]["command"] += [
                f"verl-{name}.yaml",
                "--bind",
                "reward.provider=target_length",
            ]
        payload["commands"][offset]["command"] += ["--dry-run"]
        payload["commands"][offset + 4]["command"] += ["--resume-from", "checkpoint"]
        payload["commands"][offset + 8]["command"] += ["doctor", "bundle"]
    return payload, qualification


def test_complete_direct_candidate():
    validate_direct_evidence(*evidence())


@pytest.mark.parametrize(
    "key,value",
    [
        ("rehearsal", True),
        ("source_dirty", True),
        ("installed_wheel_verified", False),
        ("wheel_sha256", "e" * 64),
        ("source_commit", "f" * 40),
    ],
)
def test_direct_candidate_identity(key, value):
    payload, qualification = evidence()
    payload[key] = value
    with pytest.raises(ValueError):
        validate_direct_evidence(payload, qualification)


@pytest.mark.parametrize(
    "key,value",
    [
        ("source_config_sha256", "a" * 64),
        ("field_rules_sha256", "b" * 64),
        ("manual_native_yaml_edits", True),
        ("runtime_bindings", []),
        ("exact_tensor_resume", False),
        ("launchable", True),
        ("distributed_execution_tested", True),
        ("ir_sha256", None),
        ("tensor_hashes", {}),
        ("updates", {"actor_records": 2}),
        ("schedule", {"cycles": 1}),
        ("memory", {"status": "measured", "peak_reserved_gib": float("nan")}),
    ],
)
def test_direct_case_is_complete(key, value):
    payload, qualification = evidence()
    payload["cases"]["grpo"][key] = value
    with pytest.raises(ValueError):
        validate_direct_evidence(payload, qualification)


def test_direct_command_cannot_be_replaced_by_native_train():
    payload, qualification = evidence()
    payload["commands"][2]["command"][1] = "train"
    with pytest.raises(ValueError, match="transcript"):
        validate_direct_evidence(payload, qualification)
