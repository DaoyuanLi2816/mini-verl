"""Execute narrow upstream primitives without importing distributed verl."""

from __future__ import annotations

import ast
import os
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
pytestmark = [pytest.mark.torch, pytest.mark.verl_conformance]


def official(path, name, namespace):
    from miniverl.algorithms.contract import UPSTREAM_VERL_COMMIT

    root = Path(os.environ.get("MINIVERL_VERL_V09_SOURCE", ".audit/upstream-verl-v0.9.0"))
    if not (root / ".git").exists():
        pytest.skip("pinned verl v0.9 source unavailable")
    source = subprocess.check_output(
        ["git", "show", f"{UPSTREAM_VERL_COMMIT}:{path}"], cwd=root, text=True, encoding="utf-8"
    )
    node = next(
        n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef) and n.name == name
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            node,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    exec(compile(module, f"pinned-verl:{path}", "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("seed", [42, 113])
def test_epoch_order_against_actual_upstream_iterator(shuffle, seed):
    from miniverl.training.sampling import epoch_samples

    iterator = official(
        "verl/utils/tensordict_utils.py",
        "make_iterator",
        {"torch": torch, "index_select_tensor_dict": lambda td, ids: [int(i) for i in ids]},
    )
    batches = list(iterator(NS(batch_size=[12], shape=[12]), 4, 3, seed, {"shuffle": shuffle}))
    expected = [[i for batch in batches[e * 3 : (e + 1) * 3] for i in batch] for e in range(3)]
    assert (
        list(epoch_samples(list(range(12)), minibatch=4, epochs=3, seed=seed, shuffle=shuffle))
        == expected
    )


@pytest.mark.parametrize("values", [[0, 0], [1, 1], [0, 1], [1], [1e-200, 2e-200], [-2, 3, 1]])
def test_filter_against_actual_upstream_replay_buffer(values):
    from miniverl.training.sampling import group_decision

    extra = [{"reward_extra_info": {"score": value}} for value in values]
    method = official(
        "verl/trainer/ppo/v1/replay_buffer.py",
        "_dapo_filtered_keys",
        {
            "np": np,
            "Counter": Counter,
            "defaultdict": defaultdict,
            "tq": NS(kv_batch_get=lambda **kwargs: {"extra_fields": extra}),
        },
    )
    host = NS(
        filter_groups_metric="score",
        finished_keys={"train": {"group"}},
        _dapo_classification_cache={"train": {}},
        partitions={"train": [f"group_{i}" for i in range(len(values))]},
    )
    rejected, _ = method(host, "train")
    rows = [
        NS(
            sample_index=i,
            samples_per_prompt=len(values),
            metadata={"task_reward": {"raw_reward": value}},
        )
        for i, value in enumerate(values)
    ]
    assert (group_decision(rows, "score") == "filtered_zero_variance") == bool(rejected)
