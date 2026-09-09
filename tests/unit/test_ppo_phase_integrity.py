"""PPO phase guards bind cached values before any mutable optimization."""

from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from miniverl.errors import LifecycleError
from miniverl.training.ppo import PPOPhaseRuntime


@pytest.mark.torch
def test_critic_scores_only_model_tokens_across_tool_observations():
    import torch

    from miniverl.config.models import MemoryStrategy

    critic = Mock()
    critic.values_at.return_value = torch.arange(4, dtype=torch.float32)
    host = NS(critic=critic, plan=NS(strategy=MemoryStrategy.RESIDENT), critic_update_count=7)
    trajectory = NS(
        token_ids=list(range(9)),
        model_generated_mask=[False, False, True, True, False, False, False, True, True],
        metadata={},
        policy_version=3,
    )
    PPOPhaseRuntime(host).score_old_values([trajectory])
    critic.values_at.assert_called_once_with(trajectory.token_ids, [1, 2, 6, 7], with_grad=False)
    assert trajectory.metadata["critic_old_values"] == [0, 1, 2, 3]
    assert trajectory.metadata["critic_provenance"]["version"] == 7


@pytest.mark.torch
def test_empty_response_cannot_receive_critic_values():
    from miniverl.config.models import MemoryStrategy
    from miniverl.errors import ConfigError

    critic = Mock()
    host = NS(critic=critic, plan=NS(strategy=MemoryStrategy.RESIDENT))
    with pytest.raises(ConfigError, match="non-empty"):
        PPOPhaseRuntime(host).score_old_values([NS(model_generated_mask=[False])])
    critic.values_at.assert_not_called()


@pytest.mark.torch
@pytest.mark.parametrize("recorded,current", [(2, 3), (3, 4), (None, 3), ("3", 3)])
def test_rl_rejects_stale_or_unbound_behavior_logprobs(recorded, current):
    from miniverl.trainer import OPDTrainer

    trajectory = NS(policy_version=3, metadata={"actor_rollout_policy_version": recorded})
    host = NS(parameter_version=current)
    with pytest.raises(LifecycleError, match="behavior"):
        OPDTrainer._build_rl_samples(host, [trajectory])


@pytest.mark.parametrize(
    "field,value",
    [
        ("identity", {"model": "wrong"}),
        ("version", 0),
        ("version", "1"),
        ("rollout_policy_version", 2),
    ],
)
def test_old_values_require_exact_critic_and_rollout_identity(field, value):
    identity = {"model": "critic"}
    host = NS(critic=NS(identity=lambda: identity), critic_update_count=1, parameter_version=3)
    provenance = {"identity": identity, "version": 1, "rollout_policy_version": 3}
    provenance[field] = value
    sample = NS(trajectory=NS(policy_version=3, metadata={"critic_provenance": provenance}))
    with pytest.raises(LifecycleError, match=r"critic|rollout"):
        PPOPhaseRuntime(host).optimize([sample])


@pytest.mark.torch
def test_critic_oom_retry_restores_rng_and_preserves_group():
    import torch

    host = NS(
        config=NS(train=NS(trajectory_batch_size=2), memory=NS(oom_retries=1)),
        critic_optimizer=Mock(),
    )
    runtime = PPOPhaseRuntime(host)
    group = [object(), object()]
    draws = []

    def compute(samples, *, physical_cap):
        assert samples is group
        draws.append(torch.rand(3))
        if physical_cap == 2:
            raise RuntimeError("CUDA out of memory after partial backward")
        return {"ok": True}

    runtime.compute_group_gradients = compute
    assert runtime._compute_with_oom_retry(group) == {"ok": True}
    assert torch.equal(draws[0], draws[1])


@pytest.mark.torch
def test_nonfinite_critic_gradient_never_commits():
    import torch

    parameter = torch.nn.Parameter(torch.ones(1))
    parameter.grad = torch.tensor([float("nan")])
    optimizer = Mock(param_groups=[{}])
    host = NS(
        critic=NS(trainable_parameters=lambda: [parameter]),
        critic_optimizer=optimizer,
        critic_schedule=NS(lr_at=lambda _: 0.001),
        critic_update_count=0,
        config=NS(critic=NS(max_grad_norm=1.0)),
    )
    with pytest.raises(LifecycleError, match="finite"):
        PPOPhaseRuntime(host).commit_update()
    optimizer.step.assert_not_called()
    assert host.critic_update_count == 0
