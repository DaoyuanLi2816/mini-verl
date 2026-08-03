from __future__ import annotations

from miniverl.environments import make_environment, make_splits
from miniverl.environments.base import OracleActionKind, ToolCall


def test_tool_policy_splits_are_deterministic_disjoint_and_cover_every_policy_dimension() -> None:
    env = make_environment("tool_policy", protocol_version="v2")
    first = make_splits(env, counts={"train": 24, "eval": 12, "test": 12}, seed=17)
    second = make_splits(env, counts={"train": 24, "eval": 12, "test": 12}, seed=17)
    assert first == second
    prompts = [task.prompt for tasks in first.values() for task in tasks]
    assert len(prompts) == len(set(prompts))
    categories = {task.metadata["policy_category"] for tasks in first.values() for task in tasks}
    assert categories == {
        "authorization",
        "confirmation",
        "instruction_hierarchy",
        "secret_exclusion",
        "benign_completion",
        "safe_error_recovery",
    }


def test_tool_policy_never_executes_an_unauthorized_or_secret_bearing_action() -> None:
    env = make_environment("tool_policy", protocol_version="v2")
    unauthorized = next(
        env.generate_task(i, 5, difficulty="easy", split="eval")
        for i in range(20)
        if env.generate_task(i, 5, difficulty="easy", split="eval").metadata["policy_category"]
        == "authorization"
    )
    env.reset(unauthorized)
    blocked = env.step(ToolCall(name="sandbox_action", arguments={"action": "delete_record"}))
    assert blocked.ok is False
    assert blocked.tool_result_metadata["executed"] is False

    secret_task = next(
        env.generate_task(i, 7, difficulty="easy", split="eval")
        for i in range(20)
        if env.generate_task(i, 7, difficulty="easy", split="eval").metadata["policy_category"]
        == "secret_exclusion"
    )
    env.reset(secret_task)
    blocked_secret = env.step(
        ToolCall(
            name="sandbox_action",
            arguments={"action": "summarize", "secret": "sandbox-secret"},
        )
    )
    assert blocked_secret.ok is False
    assert blocked_secret.tool_result_metadata["executed"] is False


def test_tool_policy_oracles_are_verifier_valid_and_recovery_is_explicit() -> None:
    env = make_environment("tool_policy", protocol_version="v2")
    for index in range(12):
        task = env.generate_task(index, 11, difficulty="easy", split="train")
        actions = env.oracle_actions(task)
        env.reset(task)
        for action in actions:
            if action.kind is OracleActionKind.TOOL_CALL:
                result = env.step(ToolCall(name=action.tool_name or "", arguments=action.arguments))
                if task.metadata["policy_category"] != "safe_error_recovery" or result.ok:
                    assert result.ok
            else:
                assert env.verify(action.answer or "").solved
        if task.metadata["policy_category"] == "safe_error_recovery":
            assert sum(a.kind is OracleActionKind.TOOL_CALL for a in actions) == 2


def test_tool_policy_privileged_context_contains_policy_but_student_prompt_does_not() -> None:
    env = make_environment("tool_policy", protocol_version="v2")
    task = env.generate_task(0, 13, difficulty="easy", split="train")
    hidden = env.privileged_context(task)
    assert hidden is not None
    assert "PRIVATE POLICY RUBRIC" in hidden
    assert "PRIVATE POLICY RUBRIC" not in env.reset(task).text
