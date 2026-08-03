"""The local role graph has verl-style semantics without distributed APIs."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def test_local_role_graph_names_every_runtime_boundary() -> None:
    from miniverl.runtime.roles import LocalRoleGraph

    values = [SimpleNamespace(name=name) for name in range(9)]
    graph = LocalRoleGraph(
        actor_policy=values[0],
        rollout_runtime=values[1],
        teacher_policy=values[2],
        reference_policy=values[3],
        reward_or_verifier=values[4],
        target_builder=values[5],
        update_runtime=values[6],
        evaluation_runtime=values[7],
        artifact_bridge=values[8],
    )

    assert graph.describe()["roles"] == {
        "actor_policy": "SimpleNamespace",
        "rollout_runtime": "SimpleNamespace",
        "teacher_policy": "SimpleNamespace",
        "reference_policy": "SimpleNamespace",
        "reward_or_verifier": "SimpleNamespace",
        "target_builder": "SimpleNamespace",
        "update_runtime": "SimpleNamespace",
        "evaluation_runtime": "SimpleNamespace",
        "artifact_bridge": "SimpleNamespace",
    }
    assert graph.describe()["execution_model"] == "local_single_process"


def test_teacher_and_reference_are_semantically_distinct_roles() -> None:
    from miniverl.runtime.roles import LocalRoleGraph

    shared_object = object()
    with pytest.raises(ValueError, match="distinct role views"):
        LocalRoleGraph(
            actor_policy=object(),
            rollout_runtime=object(),
            teacher_policy=shared_object,
            reference_policy=shared_object,
            reward_or_verifier=object(),
            target_builder=object(),
            update_runtime=object(),
            evaluation_runtime=object(),
            artifact_bridge=object(),
        )


def test_artifact_bridge_exposes_only_local_run_paths(tmp_path) -> None:
    from miniverl.runtime.roles import LocalArtifactBridge

    bridge = LocalArtifactBridge(run_root=tmp_path)
    assert bridge.describe() == {
        "kind": "local_filesystem",
        "run_root": str(tmp_path.resolve()),
    }
