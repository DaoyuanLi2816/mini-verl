"""Release gate for the exact-wheel installed PPO/GRPO command workflow."""

from __future__ import annotations

import math
import re
from typing import Any

from miniverl.bridge.rl_v09 import VERL_RL_V09_PRODUCT_PROFILE, rl_v09_field_rules_digest
from miniverl.qualification import GPUQualification

PRODUCT_CHECKS = (
    "v014_installed_ppo_grpo_workflows",
    "v014_exact_resume_and_handoff",
    "v014_inspection_integrity",
)

PRODUCT_COMMANDS = ["data"] + [
    "import-verl",
    "validate",
    "train",
    "train",
    "inspect",
    "inspect",
    "report",
    "train",
    "inspect",
    "export-adapter",
    "export-verl",
    "bridge",
] * 2


def validate_product_evidence(payload: dict[str, Any], qualification: GPUQualification) -> None:
    """Reject rehearsals, mismatched candidates and incomplete workflow records."""
    expected = {
        "schema_version": 1,
        "kind": "installed_ppo_grpo_product_workflows",
        "status": "passed",
        "source_commit": qualification.source_commit,
        "miniverl_version": qualification.miniverl_version,
        "wheel_sha256": qualification.wheel.sha256,
        "installed_wheel_verified": True,
        "source_dirty": False,
        "rehearsal": False,
    }
    for key, value in expected.items():
        if type(payload.get(key)) is not type(value) or payload[key] != value:
            raise ValueError(f"v0.14 product evidence mismatches {key}")
    environment = payload.get("environment") or {}
    if environment.get("gpu_name") != qualification.environment.gpu_name:
        raise ValueError("v0.14 product evidence GPU mismatch")
    if environment.get("torch") != qualification.environment.packages.get("torch"):
        raise ValueError("v0.14 product evidence Torch mismatch")
    if set(payload.get("cases") or {}) != {"ppo", "grpo"}:
        raise ValueError("v0.14 product evidence requires both PPO and GRPO")
    for name, count in (("ppo", 4), ("grpo", 2)):
        case = payload["cases"][name]
        if case.get("profile") != VERL_RL_V09_PRODUCT_PROFILE or case.get(
            "field_rules_sha256"
        ) != rl_v09_field_rules_digest(VERL_RL_V09_PRODUCT_PROFILE):
            raise ValueError(f"v0.14 {name}: compiler identity mismatch")
        for key, expected_value in (
            ("exact_tensor_resume", True),
            ("resume_counts_verified", True),
            ("launchable", False),
            ("distributed_execution_tested", False),
        ):
            if case.get(key) is not expected_value:
                raise ValueError(f"v0.14 {name}: missing {key}")
        if case.get("import_status") != "accepted" or case.get("handoff_verdict") != "ok":
            raise ValueError(f"v0.14 {name}: import or handoff failed")
        for key, expected_count in (
            ("actor_records", count),
            ("critic_records", count if name == "ppo" else 0),
        ):
            if (
                type(case.get("updates", {}).get(key)) is not int
                or case["updates"][key] != expected_count
            ):
                raise ValueError(f"v0.14 {name}: incorrect {key}")
        if (
            case.get("rollouts", {}).get("trajectories") != 8
            or case.get("rewards", {}).get("count") != 8
        ):
            raise ValueError(f"v0.14 {name}: incomplete rollout/reward evidence")
        memory = case.get("memory") or {}
        peak = memory.get("peak_reserved_gib")
        if (
            memory.get("status") != "measured"
            or not isinstance(peak, (int, float))
            or isinstance(peak, bool)
            or not math.isfinite(peak)
            or not 0 < peak < 14.5
        ):
            raise ValueError(f"v0.14 {name}: invalid memory measurement")
        required = {"adapter.safetensors", "optimizer.safetensors"}
        if name == "ppo":
            required |= {"critic.safetensors", "critic-optimizer.safetensors"}
        hashes = case.get("tensor_hashes") or {}
        if not required.issubset(hashes) or any(
            not isinstance(value, str) or not re.fullmatch("[0-9a-f]{64}", value)
            for value in hashes.values()
        ):
            raise ValueError(f"v0.14 {name}: missing tensor replay hashes")
    commands = payload.get("commands") or []
    if len(commands) != len(PRODUCT_COMMANDS) or any(
        type(row.get("exit_code")) is not int
        or row["exit_code"] != 0
        or row.get("command", [])[:2] != ["miniverl", expected_command]
        for row, expected_command in zip(commands, PRODUCT_COMMANDS, strict=False)
    ):
        raise ValueError("v0.14 product command transcript is incomplete")
    for offset, name in ((1, "ppo"), (13, "grpo")):
        if (
            commands[offset]["command"][2:6]
            != ["--profile", VERL_RL_V09_PRODUCT_PROFILE, "--example", name]
            or "--dry-run" not in commands[offset + 2]["command"]
            or "--resume-from" not in commands[offset + 7]["command"]
            or commands[offset + 11]["command"][2] != "doctor"
        ):
            raise ValueError(f"v0.14 {name}: transcript omits plan, resume or handoff")
