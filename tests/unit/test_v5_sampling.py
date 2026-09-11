"""Sampling order and group decisions are mathematical, not packing choices."""

from types import SimpleNamespace as NS

import pytest

from miniverl.errors import ConfigError


@pytest.mark.torch
def test_epoch_order_matches_upstream_dataloader_and_preserves_global_rng():
    import torch
    from torch.utils.data import DataLoader

    from miniverl.training.sampling import epoch_samples

    generator = torch.Generator().manual_seed(42)
    loader = DataLoader(
        torch.arange(12),
        batch_size=4,
        shuffle=True,
        generator=generator,
        collate_fn=lambda items: items,
    )
    expected = [[int(i) for batch in loader for i in batch] for _ in range(3)]
    before = torch.get_rng_state().clone()
    actual = list(epoch_samples(list(range(12)), minibatch=4, epochs=3, shuffle=True, seed=42))
    assert actual == expected
    assert actual[0] != actual[1]
    assert torch.equal(before, torch.get_rng_state())
    assert actual == list(
        epoch_samples(list(range(12)), minibatch=4, epochs=3, shuffle=True, seed=42)
    )
    assert actual != list(
        epoch_samples(list(range(12)), minibatch=4, epochs=3, shuffle=True, seed=43)
    )


@pytest.mark.torch
def test_group_filter_uses_named_raw_metric_and_requires_complete_groups():
    from miniverl.training.sampling import group_decision

    def group(values):
        return [
            NS(
                sample_index=i,
                samples_per_prompt=len(values),
                metadata={
                    "task_reward": {
                        "raw_reward": float(i),
                        "components": [{"name": "accuracy", "value": value}],
                    }
                },
            )
            for i, value in enumerate(values)
        ]

    assert group_decision(group([1, 1]), "accuracy") == "filtered_zero_variance"
    assert group_decision(group([0, 1]), "accuracy") == "retained"
    assert group_decision(group([1]), "accuracy") == "retained"
    with pytest.raises(ConfigError, match="metric"):
        group_decision(group([1, 2]), "missing")
    broken = group([0, 1])[:1]
    with pytest.raises(ConfigError, match="incomplete"):
        group_decision(broken, "accuracy")


def test_v5_accepts_shuffle_filter_and_refill_without_changing_v4(tmp_path):
    from pathlib import Path

    import yaml

    from miniverl.bridge.direct import compile_direct as compile_v4
    from miniverl.bridge.direct_v5 import compile_direct

    source = Path(__file__).resolve().parents[2] / "src/miniverl/resources/verl_direct_grpo.yaml"
    value = yaml.safe_load(source.read_text())
    value["actor_rollout_ref"]["actor"].update(shuffle=True, data_loader_seed=113)
    value["algorithm"]["filter_groups"] = {
        "enable": True,
        "metric": "score",
        "max_num_gen_batches": 2,
    }
    value["trainer"]["v1"] = {"sampler": {"sync_refill_failed_groups": True}}
    path = tmp_path / "filtered.yaml"
    path.write_text(yaml.safe_dump(value))
    assert compile_v4(path)["semantic_status"] == "rejected"
    report = compile_direct(path)
    assert report["semantic_status"] == "accepted", report["rejections"]
    assert report["sampling_semantics"]["actor_seed"] == 113
    assert report["sampling_semantics"]["refill_failed_groups"] is True
