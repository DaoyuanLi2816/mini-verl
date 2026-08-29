"""Matched interruption/resume for transactional n>1 prompt groups."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tests.conftest import requires_torch

pytestmark = [requires_torch, pytest.mark.torch]


def _config(tmp_path: Path, train: Path):  # type: ignore[no-untyped-def]
    from miniverl.config import RunConfig

    return RunConfig.model_validate(
        {
            "run": {
                "name": "grouped-resume",
                "mode": "opd",
                "seed": 117,
                "output_dir": str(tmp_path / "runs"),
                "execution_plan_digest": "d" * 64,
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
                "backend": "hf_cached",
                "samples_per_prompt": 4,
                "max_turns": 1,
                "max_new_tokens_per_turn": 3,
                "max_total_tokens": 64,
                "temperature": 0.8,
                "prompt_batch_size": 2,
                "max_padded_tokens": 128,
                "record_logprobs": True,
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
                "cycles": 2,
                "rollouts_per_cycle": 2,
                "gradient_accumulation_steps": 2,
                "learning_rate": 0.001,
            },
            "memory": {"strategy": "resident"},
            "cache": {"entries_per_shard": 4, "dtype": "float32", "keep_cycles": 2},
            "eval": {"enabled": False},
            "report": {"enabled": False},
        }
    )


def _trajectory_payloads(run_dir: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (run_dir / "trajectories.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def _cache_tensors(run_dir: Path):  # type: ignore[no-untyped-def]
    from safetensors.torch import load_file

    cache = run_dir / "teacher-cache"
    return {
        shard.name: load_file(str(shard)) for shard in sorted(cache.glob("shard-*.safetensors"))
    }


def test_interrupted_group_resume_matches_uninterrupted_run_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from miniverl.trainer import OPDTrainer

    train = tmp_path / "train.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"prompt": "alpha", "data_source": "unit"},
                {"prompt": "beta", "data_source": "unit"},
            ]
        ),
        train,
        row_group_size=1,
    )
    config = _config(tmp_path, train)

    reference = OPDTrainer.from_config(config, run_id="group-reference")
    try:
        reference_result = reference.train()
    finally:
        reference.close()

    interrupted = OPDTrainer.from_config(config, run_id="group-interrupted")
    original_generate = interrupted.student.generate_batch_cached
    calls = 0

    def interrupt_second_batch(prefixes, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt
        return original_generate(prefixes, **kwargs)

    monkeypatch.setattr(interrupted.student, "generate_batch_cached", interrupt_second_batch)
    interrupted_root = interrupted.paths.root
    with pytest.raises(KeyboardInterrupt):
        interrupted.train()
    interruption_state = json.loads(
        (interrupted_root / "checkpoints" / "interrupted-group" / "state.json").read_text(
            encoding="utf-8"
        )
    )
    assert interruption_state["prompt_cursor"] == 0
    assert interruption_state["rollout_group_cursor"] == 0
    assert interruption_state["trajectory_count"] == 0
    assert interruption_state["pending_group_identity"] == []
    interrupted.close()

    resumed = OPDTrainer.from_config(config, resume=interrupted_root)
    try:
        resumed_result = resumed.train()
    finally:
        resumed.close()

    reference_rows = _trajectory_payloads(reference_result.run_dir)
    resumed_rows = _trajectory_payloads(resumed_result.run_dir)
    assert resumed_rows == reference_rows
    identities = [(row["prompt_group_id"], row["sample_index"]) for row in resumed_rows]
    assert len(identities) == len(set(identities)) == 16
    assert resumed_result.global_step == reference_result.global_step == 8
    assert resumed_result.policy_version == reference_result.policy_version == 8

    reference_cache = _cache_tensors(reference_result.run_dir)
    resumed_cache = _cache_tensors(resumed_result.run_dir)
    assert reference_cache.keys() == resumed_cache.keys()
    for shard_name, reference_tensors in reference_cache.items():
        assert reference_tensors.keys() == resumed_cache[shard_name].keys()
        for tensor_name, reference_tensor in reference_tensors.items():
            assert reference_tensor.equal(resumed_cache[shard_name][tensor_name])

    for filename in ("adapter.safetensors", "optimizer.safetensors"):
        assert (reference_result.run_dir / "checkpoints" / "final" / filename).read_bytes() == (
            resumed_result.run_dir / "checkpoints" / "final" / filename
        ).read_bytes()
