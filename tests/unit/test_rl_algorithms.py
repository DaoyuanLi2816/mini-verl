from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.torch


def _fixture():  # type: ignore[no-untyped-def]
    rewards = torch.tensor([[0.0, 1.0, 0.0], [0.0, 3.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    groups = ("a", "a", "b", "b")
    return rewards, mask, groups


def test_grpo_uses_sample_std_and_dr_grpo_does_not_scale() -> None:
    from miniverl.algorithms.advantages import grpo_outcome_advantage

    rewards, mask, groups = _fixture()
    grpo = grpo_outcome_advantage(rewards, mask, groups, normalize_by_std=True)
    dr_grpo = grpo_outcome_advantage(rewards, mask, groups, normalize_by_std=False)

    # torch.std([1, 3]) is sqrt(2): upstream verl uses the sample std.
    expected_grpo = torch.tensor([-1 / 2**0.5, 1 / 2**0.5, -1 / 2**0.5, 1 / 2**0.5])
    expected_dr = torch.tensor([-1.0, 1.0, -1.0, 1.0])
    torch.testing.assert_close(grpo.sequence_advantages, expected_grpo, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(dr_grpo.sequence_advantages, expected_dr)
    torch.testing.assert_close(grpo.advantages, expected_grpo[:, None] * mask)
    assert grpo.zero_variance_groups == 0


def test_grpo_zero_variance_group_is_finite_and_zero() -> None:
    from miniverl.algorithms.advantages import grpo_outcome_advantage

    rewards = torch.tensor([[1.0], [1.0]])
    mask = torch.ones_like(rewards)
    output = grpo_outcome_advantage(rewards, mask, ("same", "same"))

    assert torch.isfinite(output.advantages).all()
    torch.testing.assert_close(output.advantages, torch.zeros_like(output.advantages))
    assert output.zero_variance_groups == 1


def test_rloo_is_leave_one_out_per_prompt() -> None:
    from miniverl.algorithms.advantages import rloo_outcome_advantage

    rewards, mask, groups = _fixture()
    output = rloo_outcome_advantage(rewards, mask, groups)

    torch.testing.assert_close(output.sequence_advantages, torch.tensor([-2.0, 2.0, -2.0, 2.0]))
    torch.testing.assert_close(output.advantages, output.sequence_advantages[:, None] * mask)


def test_group_estimators_reject_partial_or_singleton_groups() -> None:
    from miniverl.algorithms.advantages import grpo_outcome_advantage, rloo_outcome_advantage

    rewards = torch.ones((3, 2))
    mask = torch.ones_like(rewards)
    groups = ("a", "a", "b")
    with pytest.raises(ValueError, match="at least two complete samples"):
        grpo_outcome_advantage(rewards, mask, groups)
    with pytest.raises(ValueError, match="at least two complete samples"):
        rloo_outcome_advantage(rewards, mask, groups)


def test_reinforce_plus_plus_carries_return_across_masked_observation() -> None:
    from miniverl.algorithms.advantages import reinforce_plus_plus_advantage

    rewards = torch.tensor([[0.0, 5.0, 0.0, 2.0]])
    mask = torch.tensor([[1.0, 0.0, 1.0, 1.0]])
    output = reinforce_plus_plus_advantage(rewards, mask, gamma=0.5)

    # The masked observation does not terminate return propagation.
    torch.testing.assert_close(output.returns, torch.tensor([[0.5, 0.0, 1.0, 2.0]]))
    assert output.advantages[0, 1] == 0
    assert torch.isfinite(output.advantages).all()


def test_gae_masks_observations_and_returns_are_unwhitened() -> None:
    from miniverl.algorithms.advantages import gae_advantage_return

    rewards = torch.tensor([[1.0, 99.0, 2.0]])
    values = torch.tensor([[0.4, 0.7, 0.5]])
    mask = torch.tensor([[1.0, 0.0, 1.0]])
    output = gae_advantage_return(rewards, values, mask, gamma=0.9, lam=0.8)

    # Upstream leaves masked positions populated after whitening; downstream
    # policy/value reductions apply the mask.
    assert output.advantages.shape == rewards.shape
    assert output.returns.shape == rewards.shape
    assert torch.isfinite(output.advantages).all()


def test_policy_loss_matches_clipped_dual_clip_equations_and_gradient() -> None:
    from miniverl.algorithms.policy import clipped_policy_loss

    old = torch.tensor([-1.2, -0.7, -2.0])
    current = torch.tensor([-0.8, -1.1, -2.0], requires_grad=True)
    advantages = torch.tensor([1.0, -2.0, 0.5])
    mask = torch.tensor([1.0, 1.0, 0.0])
    output = clipped_policy_loss(
        current_log_probs=current,
        old_log_probs=old,
        advantages=advantages,
        response_mask=mask,
    )
    output.loss.backward()

    assert torch.isfinite(output.loss)
    assert current.grad is not None
    assert current.grad[2] == 0
    assert 0.0 <= output.metrics["pg_clipfrac"] <= 1.0


def test_value_loss_clips_around_old_values_and_honors_mask() -> None:
    from miniverl.algorithms.value import clipped_value_loss

    current = torch.tensor([0.8, -0.1, 4.0], requires_grad=True)
    old = torch.tensor([0.0, 0.0, 0.0])
    returns = torch.tensor([1.0, -1.0, 9.0])
    mask = torch.tensor([1.0, 1.0, 0.0])
    output = clipped_value_loss(current, old, returns, mask, cliprange_value=0.2)
    output.loss.backward()

    assert current.grad is not None
    assert current.grad[2] == 0
    assert output.metrics["value_clipfrac"] == pytest.approx(0.5)


def test_rl_actor_objective_adds_entropy_and_reference_kl_with_gradients() -> None:
    from miniverl.algorithms.kl import reference_kl_penalty
    from miniverl.algorithms.policy import clipped_policy_loss
    from miniverl.losses.chunked import VerlRLTargetProvider

    logits = torch.tensor([[0.2, -0.3, 0.8], [0.1, 0.4, -0.2]], requires_grad=True)
    targets = torch.tensor([2, 1])
    log_probs = torch.log_softmax(logits, dim=-1)
    current = log_probs.gather(-1, targets[:, None]).squeeze(-1)
    old = current.detach() - torch.tensor([0.1, -0.05])
    reference = current.detach() - torch.tensor([0.2, 0.15])
    advantages = torch.tensor([1.0, -0.5])
    provider = VerlRLTargetProvider(
        target_token_ids=targets,
        old_actor_log_probs=old,
        advantages=advantages,
        reference_log_probs=reference,
        actor_kl_coef=0.3,
        entropy_coeff=0.2,
    )

    actual = provider.divergence(0, 2, logits)
    policy = clipped_policy_loss(
        current_log_probs=current,
        old_log_probs=old,
        advantages=advantages,
        response_mask=torch.ones(2),
    ).per_token_loss
    entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
    expected = policy - 0.2 * entropy + 0.3 * reference_kl_penalty(current, reference)

    torch.testing.assert_close(actual, expected)
    assert provider.diagnostics[0]["metric_weight"] == 2.0
    actual.mean().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize("penalty", ["kl", "abs", "mse", "low_var_kl"])
def test_reference_kl_estimators_are_finite_and_shape_preserving(penalty: str) -> None:
    from miniverl.algorithms.kl import reference_kl_penalty

    old = torch.tensor([[-1.0, -0.5], [-4.0, -2.0]])
    reference = torch.tensor([[-1.2, -0.2], [-3.0, -2.5]])
    result = reference_kl_penalty(old, reference, penalty=penalty)

    assert result.shape == old.shape
    assert torch.isfinite(result).all()
