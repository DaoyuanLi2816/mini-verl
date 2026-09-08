from __future__ import annotations

import pytest

from tests.conftest import requires_torch

pytestmark = [requires_torch, pytest.mark.torch]


def test_value_critic_has_independent_trainable_state_and_round_trips(toy_tokenizer) -> None:
    import torch

    from miniverl.models.critic import ValueCritic
    from miniverl.models.toy import ToyBackend

    backbone = ToyBackend(
        tokenizer=toy_tokenizer,
        model_id="critic",
        hidden_size=16,
        num_layers=1,
        num_heads=2,
        intermediate_size=32,
        seed=7,
        device="cpu",
        trainable=True,
    )
    critic = ValueCritic(backbone, head_seed=11)
    before = critic.trainable_state_dict()
    values = critic.values_at([1, 2, 3, 4], [0, 1, 2], with_grad=True)
    values.square().mean().backward()

    assert values.shape == (3,)
    assert all(torch.isfinite(values))
    assert any(parameter.grad is not None for parameter in critic.trainable_parameters())
    assert "value_head.weight" in before
    assert "value_head.bias" in before

    with torch.no_grad():
        critic.value_head.bias.add_(1.0)
    critic.load_trainable_state_dict(before)
    restored = critic.trainable_state_dict()
    assert restored.keys() == before.keys()
    for name in before:
        assert torch.equal(restored[name], before[name]), name
    critic.release()
