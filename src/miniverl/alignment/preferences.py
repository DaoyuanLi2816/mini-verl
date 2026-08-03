"""Deterministic, harmless preference pairs for the sandbox policy suite."""

from __future__ import annotations

import hashlib
from typing import Any

from miniverl.agent.protocol import render_final, render_tool_call
from miniverl.agent.transcript import ChatFormat
from miniverl.environments.tool_policy import ToolPolicyEnvironment
from miniverl.utils.runs import canonical_json

__all__ = ["build_tool_policy_preferences", "preference_dataset_digest"]


def _prompt(environment: ToolPolicyEnvironment, task_prompt: str) -> str:
    fmt = ChatFormat()
    return (
        fmt.header("system")
        + environment.system_prompt()
        + fmt.close()
        + fmt.header("user")
        + task_prompt
        + fmt.close()
        + fmt.header("assistant")
    )


def _pair(category: str) -> tuple[str, str]:
    if category in {"authorization", "instruction_hierarchy"}:
        return (
            render_final("REFUSE"),
            render_tool_call("sandbox_action", {"action": "delete_record"}),
        )
    if category == "confirmation":
        return (
            render_final("CONFIRM"),
            render_tool_call(
                "sandbox_action",
                {"action": "archive", "confirmation": False},
            ),
        )
    if category == "secret_exclusion":
        return (
            render_tool_call("sandbox_action", {"action": "summarize", "attempt": 1}),
            render_tool_call(
                "sandbox_action",
                {
                    "action": "summarize",
                    "attempt": 1,
                    "secret": "SYNTHETIC_PRIVATE_VALUE",
                },
            ),
        )
    if category == "benign_completion":
        return (
            render_tool_call("sandbox_action", {"action": "read", "attempt": 1}),
            render_final("REFUSE"),
        )
    if category == "safe_error_recovery":
        return (
            render_tool_call("sandbox_action", {"action": "read", "attempt": 1}),
            render_final("REFUSE"),
        )
    raise ValueError(f"unknown tool-policy category {category!r}")


def build_tool_policy_preferences(
    *,
    count: int,
    seed: int,
    split: str = "train",
) -> list[dict[str, Any]]:
    """Build standard-format TRL DPO rows without executing any action."""
    if count < 1:
        raise ValueError("preference count must be positive")
    environment = ToolPolicyEnvironment(protocol_version="v2")
    rows: list[dict[str, Any]] = []
    for index in range(count):
        task = environment.generate_task(index, seed, difficulty="easy", split=split)
        category = str(task.metadata["policy_category"])
        chosen, rejected = _pair(category)
        rows.append(
            {
                "id": task.task_id,
                "prompt": _prompt(environment, task.prompt),
                "chosen": chosen,
                "rejected": rejected,
                "policy_category": category,
                "split": split,
            }
        )
    return rows


def preference_dataset_digest(rows: list[dict[str, Any]]) -> str:
    """Bind row order and exact text to one portable digest."""
    return hashlib.sha256(canonical_json(rows).encode("utf-8")).hexdigest()
