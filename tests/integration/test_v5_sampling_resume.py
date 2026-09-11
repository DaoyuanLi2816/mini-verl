"""Fault injection across filtered collection, shuffle, critic and actor phases."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("torch")
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
pytestmark = pytest.mark.torch


def sampling_config(tmp_path, algorithm="ppo"):
    from miniverl.algorithms.contract import ADVANTAGE_IMPLEMENTATION_VERSION
    from miniverl.bridge.direct_v5 import PROFILE
    from miniverl.config import RunConfig

    path = tmp_path / "train.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "prompt": f"Choose {i}",
                    "data_source": "unit",
                    "reward_model": {
                        "style": "exact",
                        "ground_truth": "drop" if i == 0 else "keep",
                    },
                }
                for i in range(8)
            ]
        ),
        path,
    )
    return RunConfig.model_validate(
        {
            "run": {
                "name": "v5-sampling",
                "mode": "rl",
                "seed": 29,
                "output_dir": str(tmp_path / "runs"),
                "profile_identity": {
                    "profile_name": PROFILE,
                    "v5_sampling": {
                        "actor_shuffle": True,
                        "actor_seed": 42,
                        "critic_shuffle": True,
                        "critic_seed": 113,
                        "filter_metric": "score",
                        "refill_failed_groups": True,
                        "max_attempted_groups": 16,
                    },
                },
            },
            "models": {
                "backend": "toy",
                "device": "cpu",
                "student": {
                    "model_id": "toy-v5",
                    "lora": {"enabled": False},
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
                "train_files": [str(path)],
                "allow_plain_string_prompts": True,
                "shuffle": False,
                "use_task_rewards": True,
            },
            "rollout": {
                "backend": "hf_cached",
                "samples_per_prompt": 2,
                "temperature": 1.0,
                "max_turns": 1,
                "max_new_tokens_per_turn": 3,
                "max_total_tokens": 64,
                "record_logprobs": True,
                "prompt_batch_size": 2,
            },
            "selection": {"selector": "all_model_tokens"},
            "loss": {
                "mode": "verl_rl_policy",
                "aggregation": "token-mean",
                "scale_by_temperature_squared": False,
                "chunk_size": 16,
            },
            "algorithm": {
                "name": algorithm,
                "implementation_version": ADVANTAGE_IMPLEMENTATION_VERSION,
                "gamma": 0.9,
                "lam": 0.8,
                "actor_ppo_epochs": 2,
            },
            "critic": {
                "enabled": algorithm == "ppo",
                "learning_rate": 0.002,
                "ppo_epochs": 2,
                "gradient_accumulation_steps": 2,
            }
            if algorithm == "ppo"
            else {},
            "reward": {"enabled": True, "provider": "python_api"},
            "train": {
                "cycles": 2,
                "rollouts_per_cycle": 2,
                "gradient_accumulation_steps": 2,
                "learning_rate": 0.001,
                "opd_freshness": "replay",
            },
            "memory": {"strategy": "resident"},
            "eval": {"enabled": False},
            "report": {"enabled": False},
        }
    )


class GroupReward:
    @property
    def identity(self):
        from miniverl.rewards import RewardProviderIdentity

        return RewardProviderIdentity(name="group-score", version="v1", config_digest="1" * 64)

    def score(self, request):
        from miniverl.rewards import RewardResult, RewardStatus

        return RewardResult(
            trajectory_id=request.trajectory_id,
            prompt_group_id=request.prompt_group_id,
            sample_index=request.sample_index,
            samples_per_prompt=request.samples_per_prompt,
            provider=self.identity,
            input_digest=request.input_digest,
            raw_reward=0.0 if request.ground_truth == "drop" else float(request.sample_index),
            status=RewardStatus.OK,
            duration_ms=0.0,
            deterministic=True,
        )


def fail_one_group(trainer):
    from miniverl.training.sampling import RolloutGroupFailure

    original = trainer.rollout_runtime.generate

    def generate(*args, **kwargs):
        if trainer._rollout_group_cursor == 1:
            raise RolloutGroupFailure("injected terminal empty group")
        return original(*args, **kwargs)

    trainer.rollout_runtime.generate = generate


def rows(root, filename):
    return [json.loads(line) for line in (root / filename).read_text().splitlines()]


@pytest.mark.parametrize(
    "algorithm,phase",
    [
        ("ppo", "collection"),
        ("ppo", "critic"),
        ("ppo", "actor"),
        ("grpo", "collection"),
        ("grpo", "actor"),
    ],
)
def test_filtered_refill_shuffled_epochs_resume_exactly(tmp_path, monkeypatch, algorithm, phase):
    from miniverl.trainer import OPDTrainer
    from miniverl.utils.seeding import seed_everything

    config = sampling_config(tmp_path, algorithm)
    seed_everything(29, deterministic=True)
    with OPDTrainer.from_config(
        config, run_id="reference", reward_provider=GroupReward()
    ) as trainer:
        fail_one_group(trainer)
        reference = trainer.train().run_dir
    seed_everything(29, deterministic=True)
    with OPDTrainer.from_config(
        config, run_id="interrupted", reward_provider=GroupReward()
    ) as trainer:
        fail_one_group(trainer)
        root = trainer.paths.root
        if phase == "collection":
            original = trainer.rollout_runtime.generate

            def interrupt(*args, **kwargs):
                if trainer._rollout_group_cursor == 2:
                    raise KeyboardInterrupt
                return original(*args, **kwargs)

            monkeypatch.setattr(trainer.rollout_runtime, "generate", interrupt)
        else:
            name = "_optimize_critic" if phase == "critic" else "_optimize"
            original = getattr(trainer, name)

            def interrupt(*args, **kwargs):
                original(*args, **kwargs)
                raise KeyboardInterrupt

            monkeypatch.setattr(trainer, name, interrupt)
        with pytest.raises(KeyboardInterrupt):
            trainer.train()
    with OPDTrainer.from_config(config, resume=root, reward_provider=GroupReward()) as trainer:
        fail_one_group(trainer)
        result = trainer.train()
    assert result.global_step == 8
    filenames = ["adapter.safetensors", "optimizer.safetensors"]
    if algorithm == "ppo":
        assert result.critic_update_count == 8
        filenames += ["critic.safetensors", "critic-optimizer.safetensors"]
    for filename in filenames:
        assert (reference / "checkpoints/final" / filename).read_bytes() == (
            root / "checkpoints/final" / filename
        ).read_bytes()
    for filename in ["trajectories.jsonl", "rewards.jsonl", "advantages.jsonl"]:
        assert rows(reference, filename) == rows(root, filename)

    def selection_records(root):
        return [
            r
            for r in rows(root, "metrics.jsonl")
            if r["phase"] in {"rl_group_selection", "rl_sampling"}
        ]

    assert selection_records(reference) == selection_records(root)
    summaries = [r for r in selection_records(root) if r["phase"] == "rl_sampling"]
    assert summaries[0]["filtered_groups"] == 1
    assert summaries[0]["failed_groups"] == 1
    assert all(r["retained_groups"] == 2 for r in summaries)

    def minibatches(root):
        return [
            (r["phase"], r["ppo_epoch"], r["minibatch_trajectory_ids"])
            for r in rows(root, "metrics.jsonl")
            if "minibatch_trajectory_ids" in r
        ]

    assert minibatches(reference) == minibatches(root)


def test_attempt_guard_never_commits_a_partial_optimizer_batch(tmp_path):
    from miniverl.errors import ConfigError
    from miniverl.trainer import OPDTrainer

    config = sampling_config(tmp_path)
    config.run.profile_identity["v5_sampling"]["max_attempted_groups"] = 1
    with OPDTrainer.from_config(config, reward_provider=GroupReward()) as trainer:
        with pytest.raises(ConfigError, match="guard exhausted"):
            trainer.train()
        assert trainer.global_step == 0
        assert not any(
            r["phase"] in {"critic_update", "update"}
            for r in rows(trainer.paths.root, "metrics.jsonl")
        )


def test_filter_surplus_and_handoff_preserve_v5_semantics(tmp_path):
    from miniverl.bridge.export import _rl_v09_overrides
    from miniverl.reporting.inspection import inspect_run
    from miniverl.trainer import OPDTrainer

    config = sampling_config(tmp_path)
    with OPDTrainer.from_config(config, reward_provider=GroupReward()) as trainer:
        root = trainer.train().run_dir
    summaries = [r for r in rows(root, "metrics.jsonl") if r["phase"] == "rl_sampling"]
    assert summaries[0]["attempted_groups"] == 4
    assert summaries[0]["discarded_surplus_groups"] == 1
    assert len(summaries[0]["selected_trajectory_ids"]) == 4
    inspected = inspect_run(root)
    assert inspected["sampling"]["selected_trajectories"] == 8
    from miniverl.errors import ReportError
    from miniverl.reporting.inspection import inspect_artifacts

    metrics = rows(root, "metrics.jsonl")
    for row in metrics:
        if row.get("decision") == "discarded_surplus_group":
            row["trajectory_ids"] = []
    with pytest.raises(ReportError, match="unaccounted"):
        inspect_artifacts(root, json.loads((root / "manifest.json").read_text()), metrics, [])
    result = _rl_v09_overrides(
        config.model_dump(mode="json"),
        {"rank": 8, "alpha": 16, "target_modules": ["q_proj"]},
        {"train": ["data/train.parquet"], "val": []},
    )
    assert result["actor_rollout_ref"]["actor"]["shuffle"] is True
    assert result["critic"]["data_loader_seed"] == 113
    assert result["algorithm"]["filter_groups"]["metric"] == "score"
    assert result["trainer"]["v1"]["sampler"]["sync_refill_failed_groups"] is True


def test_oom_repartition_keeps_filter_retry_shuffle_and_updates(tmp_path, monkeypatch):
    import torch

    from miniverl.trainer import OPDTrainer
    from miniverl.utils.seeding import seed_everything

    config = sampling_config(tmp_path)
    outputs = []
    calls = []
    for injected in (False, True):
        seed_everything(29, deterministic=True)
        with OPDTrainer.from_config(
            config, run_id=f"oom-{injected}", reward_provider=GroupReward()
        ) as trainer:
            fail_one_group(trainer)
            if injected:
                original = trainer.student.generate_batch_cached

                def repartition(prefixes, original=original, **kwargs):
                    if len(prefixes) > 1:
                        calls.append(len(prefixes))
                        raise torch.cuda.OutOfMemoryError("injected physical generation OOM")
                    return original(prefixes, **kwargs)

                monkeypatch.setattr(trainer.student, "generate_batch_cached", repartition)
            outputs.append(trainer.train().run_dir)
    assert calls
    for filename in (
        "adapter.safetensors",
        "optimizer.safetensors",
        "critic.safetensors",
        "critic-optimizer.safetensors",
    ):
        assert (outputs[0] / "checkpoints/final" / filename).read_bytes() == (
            outputs[1] / "checkpoints/final" / filename
        ).read_bytes()
    for filename in ("trajectories.jsonl", "rewards.jsonl", "advantages.jsonl"):
        assert rows(outputs[0], filename) == rows(outputs[1], filename)
    decisions = [
        [
            r
            for r in rows(root, "metrics.jsonl")
            if r["phase"] in {"rl_group_selection", "rl_sampling"}
        ]
        for root in outputs
    ]
    assert decisions[0] == decisions[1]
    assert all(
        row["prompt_group_id"]
        for row in decisions[1]
        if row.get("decision") == "replaced_failed_group"
    )
