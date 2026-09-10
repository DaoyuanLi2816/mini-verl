"""Hardware/input preflight is tested independently of training or Hub access."""

from __future__ import annotations

from pathlib import Path

import pytest

from miniverl.bridge.direct import compile_direct
from miniverl.bridge.direct_runtime import _local_snapshot_digest, prepare_direct

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.torch


class Tokenizer:
    def apply_chat_template(self, messages, **_kwargs):
        return " ".join(message["content"] for message in messages)

    def encode(self, text):
        return list(range(len(text)))


@pytest.mark.parametrize("algorithm", ["ppo", "grpo"])
@pytest.mark.parametrize("gpu", [False, True])
def test_direct_preflight_retains_batch_and_derives_filtered_epoch(
    tmp_path, monkeypatch, algorithm, gpu
):
    import torch
    import yaml

    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    data = tmp_path / "rows.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "prompt": [{"role": "user", "content": "x" * size}],
                    "data_source": "test",
                    "reward_model": {"style": "target_length", "characters": 8},
                }
                for size in [2, 3, 300, 5, 6]
            ]
        ),
        data,
    )
    source = yaml.safe_load(
        (ROOT / f"src/miniverl/resources/verl_direct_{algorithm}.yaml").read_text()
    )
    source["trainer"]["total_training_steps"] = None
    source["trainer"]["total_epochs"] = 2
    path = tmp_path / "input.yaml"
    path.write_text(yaml.safe_dump(source))
    monkeypatch.setattr(
        "miniverl.bridge.direct_runtime._snapshot", lambda *a, **kw: ("a" * 40, 600_000_000)
    )
    monkeypatch.setattr("miniverl.models.factory.build_tokenizer", lambda *a, **kw: Tokenizer())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: gpu)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (15 * 2**30, 16 * 2**30))
    report = compile_direct(
        path,
        bindings=[
            f"data.train_files={data.as_posix()}",
            f"data.val_files={data.as_posix()}",
            "reward.provider=target_length",
        ],
    )
    config, result = prepare_direct(report)
    assert result["semantic_status"] == "accepted"
    if not gpu:
        assert config is None
        assert result["execution_status"] == "hardware_blocked"
        return
    assert config is not None
    assert result["schedule"] == {"epochs": 2, "batches_per_epoch": 2, "cycles": 4}
    assert result["filtered_dataset_rows"] == {"train": 4, "val": 4}
    assert config.train.rollouts_per_cycle == 2
    assert config.train.gradient_accumulation_steps == 4
    assert config.train.trajectory_batch_size == 1
    assert config.run.execution_plan_digest
    # Free VRAM fluctuates, so it is not part of the immutable resume identity.
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (14 * 2**30, 16 * 2**30))
    repeated, _ = prepare_direct(report)
    assert repeated.run.execution_plan_digest == config.run.execution_plan_digest


def test_local_snapshot_hash_is_content_bound_not_location(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    for directory in (first, second):
        (directory / "model.safetensors").write_bytes(b"weights")
        (directory / "config.json").write_text("{}")
    assert _local_snapshot_digest(str(first)) == _local_snapshot_digest(str(second))
    (second / "model.safetensors").write_bytes(b"changed")
    assert _local_snapshot_digest(str(first)) != _local_snapshot_digest(str(second))
