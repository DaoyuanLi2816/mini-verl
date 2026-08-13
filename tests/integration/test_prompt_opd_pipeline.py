"""End-to-end pure OPD over first-class verl-style Parquet prompts."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tests.conftest import requires_torch

pytestmark = [requires_torch, pytest.mark.torch]


def test_prompt_resume_reconstructs_the_dataset_cursor(monkeypatch) -> None:
    from miniverl.config.models import VerlParquetSourceConfig
    from miniverl.trainer import OPDTrainer

    class Dataset:
        def iter_split(self, split: str, *, epoch: int) -> Any:
            assert split == "train"
            return iter([f"e{epoch}-r{index}" for index in range(4)])

    monkeypatch.setattr(
        "miniverl.data.verl_parquet.render_prompt",
        lambda record, tokenizer, source: record,
    )
    trainer = object.__new__(OPDTrainer)
    trainer.prompt_dataset = Dataset()
    trainer.prompt_dataset_manifest = SimpleNamespace(rows={"train": 4})
    trainer.config = SimpleNamespace(source=VerlParquetSourceConfig(train_files=["unused"]))
    trainer.tokenizer = object()
    trainer._prompt_train_iterator = None
    trainer._prompt_train_epoch = 0
    trainer.task_cursor = 6

    assert trainer._next_tasks(4) == ["e1-r2", "e1-r3", "e2-r0", "e2-r1"]
    assert trainer.task_cursor == 10


def test_prompt_opd_trains_without_an_environment_or_reward(tmp_path) -> None:
    from miniverl.config import RunConfig
    from miniverl.trainer import OPDTrainer

    train = tmp_path / "train.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "prompt": "Compute 1 + 1.",
                    "data_source": "unit",
                    "ability": "arithmetic",
                    "extra_info": {"row": 0},
                },
                {
                    "prompt": "Compute 2 + 2.",
                    "data_source": "unit",
                    "ability": "arithmetic",
                    "extra_info": {"row": 1},
                },
            ]
        ),
        train,
        row_group_size=1,
    )
    config = RunConfig.model_validate(
        {
            "run": {
                "name": "prompt-opd",
                "mode": "opd",
                "seed": 9,
                "output_dir": str(tmp_path / "runs"),
                "execution_plan_digest": "a" * 64,
            },
            "models": {
                "backend": "toy",
                "device": "cpu",
                "student": {
                    "model_id": "toy-student",
                    "lora": {"enabled": False},
                    "toy": {
                        "hidden_size": 16,
                        "num_layers": 1,
                        "num_heads": 2,
                        "intermediate_size": 32,
                        "max_position_embeddings": 128,
                    },
                },
                "teacher": {
                    "model_id": "toy-teacher",
                    "toy_pretrain_steps": 0,
                    "toy": {
                        "hidden_size": 16,
                        "num_layers": 1,
                        "num_heads": 2,
                        "intermediate_size": 32,
                        "max_position_embeddings": 128,
                    },
                },
            },
            "source": {
                "kind": "verl_parquet",
                "train_files": [str(train)],
                "allow_plain_string_prompts": True,
                "max_prompt_length": 32,
                "shuffle": False,
            },
            "rollout": {
                "max_turns": 1,
                "max_new_tokens_per_turn": 3,
                "max_total_tokens": 64,
                "temperature": 0.0,
                "prompt_batch_size": 2,
                "max_padded_tokens": 128,
            },
            "selection": {"selector": "all_model_tokens"},
            "loss": {
                "mode": "forward_kl_topk",
                "divergence": "forward_kl",
                "aggregation": "token-mean",
                "temperature": 1.0,
                "scale_by_temperature_squared": False,
                "top_k": 4,
                "log_prob_min_clamp": -10.0,
                "chunk_size": 16,
            },
            "train": {
                "cycles": 1,
                "rollouts_per_cycle": 2,
                "gradient_accumulation_steps": 2,
                "learning_rate": 0.001,
            },
            "memory": {"strategy": "resident"},
            "cache": {"entries_per_shard": 2, "dtype": "float32"},
            "eval": {"enabled": False},
            "report": {"enabled": False},
        }
    )

    trainer = OPDTrainer.from_config(config, run_id="prompt-opd-test")
    try:
        result = trainer.train()
    finally:
        trainer.close()

    assert result.global_step == 1
    assert result.policy_version == 1
    manifest = json.loads((result.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source"]["kind"] == "verl_parquet"
    assert manifest["execution_plan_digest"] == "a" * 64
    assert manifest["source"]["rows"] == {"train": 2, "val": 0}
    cache_index = json.loads(
        (result.run_dir / "teacher-cache" / "index.json").read_text(encoding="utf-8")
    )
    assert cache_index["execution_plan_digest"] == "a" * 64
    checkpoint_state = json.loads(
        (result.run_dir / "checkpoints" / "final" / "state.json").read_text(encoding="utf-8")
    )
    assert checkpoint_state["execution_plan_digest"] == "a" * 64
    rows = [
        json.loads(line)
        for line in (result.run_dir / "trajectories.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 2
    assert all(
        not any(row["model_generated_mask"][: row["metadata"]["prompt_token_count"]])
        for row in rows
    )
    assert all(row["metadata"]["reward_model"] is None for row in rows)
    metrics = [
        json.loads(line)
        for line in (result.run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    update = next(row for row in metrics if row.get("phase") == "opd")
    cycle = next(row for row in metrics if row.get("phase") == "opd_cycle")
    assert cycle["rollout_execution"] == {
        "physical_batch_sizes": [2],
        "oom_downshifts": 0,
    }
    assert update["loss_aggregation"] == "token-mean"
    assert set(update["verl_forward_kl_topk"]) == {
        "student_mass_mean",
        "student_mass_min",
        "student_mass_max",
        "teacher_mass_mean",
        "teacher_mass_min",
        "teacher_mass_max",
        "overlap_ratio",
        "overlap_token_advantage",
    }
