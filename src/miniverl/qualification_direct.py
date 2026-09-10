"""Exact-candidate release gate for zero-rewrite direct verl execution."""

from __future__ import annotations

import hashlib
import math
import re
from importlib.resources import files
from typing import Any

from miniverl.bridge.direct import DIRECT_PROFILE, direct_rules_digest
from miniverl.qualification import GPUQualification

DIRECT_CHECKS = (
    "v015_direct_upstream_config",
    "v015_exact_resume_handoff",
    "v015_no_native_yaml_edits",
)
DIRECT_COMMANDS = ["data"] + [
    "run",
    "run",
    "inspect",
    "report",
    "run",
    "inspect",
    "export-adapter",
    "export-verl",
    "bridge",
] * 2


def validate_direct_evidence(payload: dict[str, Any], qualification: GPUQualification) -> None:
    expected = {
        "schema_version": 1,
        "kind": "installed_direct_verl_workflows",
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
            raise ValueError(f"v0.15 direct evidence mismatches {key}")
    environment = payload.get("environment") or {}
    if environment.get("gpu_name") != qualification.environment.gpu_name or environment.get(
        "torch"
    ) != qualification.environment.packages.get("torch"):
        raise ValueError("v0.15 direct GPU/Torch identity mismatch")
    if set(payload.get("cases") or {}) != {"ppo", "grpo"}:
        raise ValueError("v0.15 direct requires PPO and GRPO")
    for name in ("ppo", "grpo"):
        case = payload["cases"][name]
        required = {
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
        }
        for key, value in required.items():
            if type(case.get(key)) is not type(value) or case[key] != value:
                raise ValueError(f"v0.15 {name}: invalid {key}")
        for field, value in (("actor_records", 4), ("critic_records", 4 if name == "ppo" else 0)):
            if (
                type(case.get("updates", {}).get(field)) is not int
                or case["updates"][field] != value
            ):
                raise ValueError(f"v0.15 {name}: incorrect {field}")
        if (
            case.get("rollouts", {}).get("trajectories") != 8
            or case.get("rewards", {}).get("count") != 8
        ):
            raise ValueError("v0.15 incomplete direct sample/reward evidence")
        if (
            case.get("schedule", {}).get("cycles") != 2
            or case.get("hardware_plan", {}).get("status") != "executable_with_local_lowering"
        ):
            raise ValueError("v0.15 incomplete direct schedule/hardware plan")
        bindings = case.get("runtime_bindings") or []
        if (
            len(bindings) != 1
            or bindings[0].get("binding") != "reward.provider"
            or bindings[0].get("bound") != "target_length"
        ):
            raise ValueError("v0.15 missing explicit qualification reward binding")
        memory = case.get("memory") or {}
        peak = memory.get("peak_reserved_gib")
        if (
            memory.get("status") != "measured"
            or not isinstance(peak, (int, float))
            or isinstance(peak, bool)
            or not math.isfinite(peak)
            or not 0 < peak < 14.5
        ):
            raise ValueError("v0.15 invalid memory evidence")
        hashes = case.get("tensor_hashes") or {}
        required_hashes = {"adapter.safetensors", "optimizer.safetensors"}
        if name == "ppo":
            required_hashes |= {"critic.safetensors", "critic-optimizer.safetensors"}
        if not required_hashes.issubset(hashes) or any(
            not isinstance(value, str) or not re.fullmatch("[0-9a-f]{64}", value)
            for value in [*hashes.values(), case.get("ir_sha256")]
        ):
            raise ValueError("v0.15 missing exact IR/tensor hashes")
    commands = payload.get("commands") or []
    if len(commands) != len(DIRECT_COMMANDS) or any(
        type(row.get("exit_code")) is not int
        or row["exit_code"] != 0
        or row.get("command", [])[:2] != ["miniverl", expected_command]
        for row, expected_command in zip(commands, DIRECT_COMMANDS, strict=False)
    ):
        raise ValueError("v0.15 direct command transcript incomplete")
    for offset, name in ((1, "ppo"), (10, "grpo")):
        for index in (offset, offset + 1, offset + 4):
            command = commands[index]["command"]
            if command[2:5] != [f"verl-{name}.yaml", "--bind", "reward.provider=target_length"]:
                raise ValueError(
                    "v0.15 command did not directly consume upstream source with explicit bindings"
                )
        if (
            "--dry-run" not in commands[offset]["command"]
            or "--resume-from" not in commands[offset + 4]["command"]
            or commands[offset + 8]["command"][2:3] != ["doctor"]
        ):
            raise ValueError("v0.15 transcript omits direct plan, resume or doctor")
