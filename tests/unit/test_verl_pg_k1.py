from __future__ import annotations

import pytest
import torch

from miniverl.losses.verl_pg import (
    VERL_PG_K1_IMPLEMENTATION_VERSION,
    verl_k1_estimator,
    verl_pg_k1_loss,
)


def test_k1_estimator_and_detached_advantage_match_pinned_verl() -> None:
    old = torch.tensor([-2.0, -0.5, -4.0], requires_grad=True)
    teacher = torch.tensor([-1.0, -1.5, -2.5], requires_grad=True)

    estimator = verl_k1_estimator(old, teacher)
    output = verl_pg_k1_loss(
        current_log_probs=old,
        old_log_probs=old,
        teacher_log_probs=teacher,
        weights=torch.ones(3),
    )

    torch.testing.assert_close(estimator, old - teacher)
    torch.testing.assert_close(output.advantages, (teacher - old).detach())
    assert output.advantages.requires_grad is False
    assert VERL_PG_K1_IMPLEMENTATION_VERSION == "verl-v0.8-pg-k1-v1"


def test_vanilla_policy_loss_matches_pinned_verl_equations_and_metrics() -> None:
    current = torch.tensor([-1.8, -0.9, -4.4], requires_grad=True)
    old = torch.tensor([-2.0, -0.5, -4.0])
    teacher = torch.tensor([-1.0, -1.5, -2.5])
    weights = torch.tensor([1.0, 0.0, 1.0])

    output = verl_pg_k1_loss(
        current_log_probs=current,
        old_log_probs=old,
        teacher_log_probs=teacher,
        weights=weights,
        clip_ratio=0.2,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        clip_ratio_c=3.0,
    )

    advantage = (teacher - old).detach()
    negative_approx_kl = (current - old).clamp(-20.0, 20.0)
    ratio = negative_approx_kl.exp()
    loss1 = -advantage * ratio
    loss2 = -advantage * ratio.clamp(0.8, 1.2)
    clipped = torch.maximum(loss1, loss2)
    expected_tokens = torch.where(advantage < 0, torch.minimum(-advantage * 3.0, clipped), clipped)
    expected = (expected_tokens * weights).sum() / weights.sum()

    torch.testing.assert_close(output.per_token_loss, expected_tokens)
    torch.testing.assert_close(output.loss, expected)
    assert output.metrics["pg_clipfrac"] == pytest.approx(0.5)
    assert output.metrics["ppo_kl"] == pytest.approx(0.1)
    assert output.metrics["pg_clipfrac_lower"] == pytest.approx(0.0)


def test_teacher_preference_changes_actor_gradient_in_expected_direction() -> None:
    logits = torch.tensor([[0.2, -0.1, 0.0]], requires_grad=True)
    sampled = torch.tensor([1])
    current = torch.log_softmax(logits, dim=-1).gather(-1, sampled[:, None]).squeeze(-1)
    old = current.detach()

    favored = verl_pg_k1_loss(
        current_log_probs=current,
        old_log_probs=old,
        teacher_log_probs=old + 1.0,
        weights=torch.ones(1),
    )
    favored.loss.backward()
    favored_grad = logits.grad.detach().clone()

    logits.grad = None
    current = torch.log_softmax(logits, dim=-1).gather(-1, sampled[:, None]).squeeze(-1)
    disfavored = verl_pg_k1_loss(
        current_log_probs=current,
        old_log_probs=old,
        teacher_log_probs=old - 1.0,
        weights=torch.ones(1),
    )
    disfavored.loss.backward()

    assert favored_grad[0, 1] < 0.0
    assert logits.grad is not None
    assert logits.grad[0, 1] > 0.0
    torch.testing.assert_close(favored_grad, -logits.grad)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_pg_loss_rejects_nonfinite_inputs(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        verl_pg_k1_loss(
            current_log_probs=torch.tensor([value]),
            old_log_probs=torch.tensor([-1.0]),
            teacher_log_probs=torch.tensor([-1.0]),
            weights=torch.ones(1),
        )
