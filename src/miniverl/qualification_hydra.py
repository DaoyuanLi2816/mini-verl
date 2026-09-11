"""Exact-candidate gates for native config-tree execution and deterministic replay."""

from __future__ import annotations

import math
import re
from typing import Any

from miniverl.bridge.direct_v5 import PROFILE, rules_digest
from miniverl.bridge.hydra import compose_verl
from miniverl.bridge.hydra_examples import qualification_overrides
from miniverl.qualification import GPUQualification

HYDRA_CHECKS = ("v016_native_hydra_ppo_grpo", "v016_shuffled_filtered_resume", "v016_hydra_handoff")


def validate_hydra_evidence(payload: dict[str, Any], qualification: GPUQualification) -> None:
    def require(actual: Any, expected: Any, label: str) -> None:
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(f"v0.16 Hydra evidence mismatches {label}")

    for key, value in {
        "schema_version": 1,
        "kind": "installed_native_hydra_workflows",
        "status": "passed",
        "source_commit": qualification.source_commit,
        "miniverl_version": qualification.miniverl_version,
        "wheel_sha256": qualification.wheel.sha256,
        "installed_wheel_verified": True,
        "source_dirty": False,
        "rehearsal": False,
    }.items():
        require(payload.get(key), value, key)
    require(
        payload.get("environment", {}).get("gpu_name"),
        qualification.environment.gpu_name,
        "gpu_name",
    )
    require(
        payload.get("environment", {}).get("torch"),
        qualification.environment.packages.get("torch"),
        "torch",
    )
    require(set(payload.get("cases") or {}), {"ppo", "grpo"}, "cases")
    for name, case in payload["cases"].items():
        expected = compose_verl(config_name="ppo_trainer", overrides=qualification_overrides(name))
        for key, value in {
            "profile": PROFILE,
            "field_rules_sha256": rules_digest(),
            "composition": expected.provenance,
            "composition_status": "resolved",
            "semantic_status": "accepted",
            "execution_status": "completed",
            "resolved_bytes_preserved": True,
            "manual_native_yaml_edits": False,
            "exact_tensor_resume": True,
            "exact_semantic_journal_resume": True,
            "handoff_verdict": "ok",
            "journal_normalization": "rewards.jsonl: omit top-level duration_ms only; other journals byte-exact",
            "launchable": False,
            "distributed_execution_tested": False,
        }.items():
            require(case.get(key), value, f"{name}.{key}")
        require(case.get("updates", {}).get("actor_records"), 4, f"{name}.actor_updates")
        require(
            case.get("updates", {}).get("critic_records"),
            4 if name == "ppo" else 0,
            f"{name}.critic_updates",
        )
        semantics = case.get("sampling_semantics") or {}
        require(semantics.get("actor_shuffle"), True, f"{name}.shuffle")
        require(
            semantics.get("filter_metric"), "score" if name == "grpo" else None, f"{name}.filter"
        )
        require(semantics.get("refill_failed_groups"), True, f"{name}.refill")
        require(semantics.get("max_attempted_groups"), 64, f"{name}.guard")
        bindings = {
            row.get("binding"): row.get("bound") for row in case.get("runtime_bindings", [])
        }
        require(
            bindings,
            {"reward.provider": "target_length", "sampling.max_attempted_groups": 64},
            f"{name}.bindings",
        )
        require(
            case.get("hardware_plan", {}).get("status"),
            "executable_with_local_lowering",
            f"{name}.hardware",
        )
        require(semantics.get("actor_seed"), 42, f"{name}.actor_seed")
        if name == "ppo":
            require(semantics.get("critic_shuffle"), True, "ppo.critic_shuffle")
            require(semantics.get("critic_seed"), 113, "ppo.critic_seed")
        summaries = case.get("sampling") or []
        require(len(summaries), 2, f"{name}.cycles")
        for row in summaries:
            require(row.get("retained_groups"), 2, f"{name}.retained")
            if (
                not 2 <= row.get("attempted_groups", 0) <= 64
                or len(row.get("selected_trajectory_ids", [])) != 4
            ):
                raise ValueError("v0.16 invalid group accounting")
        selected = [key for row in summaries for key in row["selected_trajectory_ids"]]
        if len(set(selected)) != 8:
            raise ValueError("v0.16 duplicated selected trajectory identity")
        require(
            len(case.get("minibatches") or []),
            8 if name == "ppo" else 4,
            f"{name}.minibatch_records",
        )
        for batch in case["minibatches"]:
            ids = batch.get("minibatch_trajectory_ids") or []
            if (
                len(ids) != 4
                or len(set(ids)) != 4
                or not set(ids).issubset(selected)
                or batch.get("ppo_epoch") not in {0, 1}
            ):
                raise ValueError("v0.16 invalid minibatch identities or epoch")
        memory = case.get("memory") or {}
        peak = memory.get("peak_reserved_gib")
        if (
            memory.get("status") != "measured"
            or isinstance(peak, bool)
            or not isinstance(peak, (int, float))
            or not math.isfinite(peak)
            or not 0 < peak < 14.5
        ):
            raise ValueError("v0.16 invalid measured memory")
        hashes = case.get("tensor_hashes") or {}
        required = {"adapter.safetensors", "optimizer.safetensors"}
        if name == "ppo":
            required |= {"critic.safetensors", "critic-optimizer.safetensors"}
        journals = case.get("journal_hashes") or {}
        if (
            not required.issubset(hashes)
            or set(journals) != {"trajectories.jsonl", "rewards.jsonl", "advantages.jsonl"}
            or any(
                not isinstance(v, str) or not re.fullmatch("[a-f0-9]{64}", v)
                for v in [*hashes.values(), *journals.values(), case.get("ir_sha256")]
            )
        ):
            raise ValueError("v0.16 missing tensor/journal/IR digests")
    commands = payload.get("commands") or []
    expected_commands = ["data"] + [
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
    if len(commands) != len(expected_commands) or any(
        row.get("exit_code") != 0 or row.get("command", [])[:2] != ["miniverl", name]
        for row, name in zip(commands, expected_commands, strict=False)
    ):
        raise ValueError("v0.16 incomplete command transcript")
    for offset, name in ((1, "ppo"), (10, "grpo")):
        expected_overrides = qualification_overrides(name)
        for index in (offset, offset + 1, offset + 4):
            command = commands[index]["command"]
            if (
                command[2:4] != ["--verl-config-name", "ppo_trainer"]
                or command[-len(expected_overrides) - 1 : -1] != expected_overrides
                or command[-1] != "--json"
            ):
                raise ValueError("v0.16 transcript did not consume native Hydra inputs")
            if (
                "reward.provider=target_length" not in command
                or "sampling.max_attempted_groups=64" not in command
            ):
                raise ValueError("v0.16 transcript omits explicit runtime bindings")
        if (
            "--dry-run" not in commands[offset]["command"]
            or "--resume-from" not in commands[offset + 4]["command"]
            or commands[offset + 8]["command"][2:3] != ["doctor"]
        ):
            raise ValueError("v0.16 transcript omits dry-run, resume or doctor")
