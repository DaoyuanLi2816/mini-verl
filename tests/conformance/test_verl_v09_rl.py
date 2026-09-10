# ruff: noqa: E402 - validate upstream source before torch-backed imports
"""Numerical and gradient conformance against pinned official verl v0.9.0."""

from __future__ import annotations

import ast
import math
import os
import subprocess
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest

torch = pytest.importorskip("torch")

from miniverl.algorithms.advantages import (
    gae_advantage_return,
    grpo_outcome_advantage,
    reinforce_plus_plus_advantage,
    rloo_outcome_advantage,
)
from miniverl.algorithms.contract import UPSTREAM_VERL_COMMIT
from miniverl.algorithms.kl import reference_kl_penalty
from miniverl.algorithms.policy import clipped_policy_loss
from miniverl.algorithms.value import clipped_value_loss

pytestmark = [pytest.mark.torch, pytest.mark.verl_conformance]


@pytest.mark.parametrize("kind", ["constant", "cosine"])
@pytest.mark.parametrize("warmup", [0, 3])
def test_direct_rollout_iteration_schedule_matches_upstream(kind, warmup):
    from miniverl.config.models import LRSchedule
    from miniverl.training.optim import LearningRateSchedule

    root = Path(os.environ.get("MINIVERL_VERL_V09_SOURCE", ".audit/upstream-verl-v0.9.0"))
    if not (root / ".git").exists():
        pytest.skip("pinned verl v0.9.0 source checkout is unavailable")
    source = subprocess.check_output(
        ["git", "show", f"{UPSTREAM_VERL_COMMIT}:verl/utils/torch_functional.py"],
        cwd=root,
        text=True,
    )
    name = f"get_{kind}_schedule_with_warmup"
    selected = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            selected,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = {"math": math, "LambdaLR": torch.optim.lr_scheduler.LambdaLR}
    exec(compile(module, "pinned-verl-torch-functional", "exec"), namespace)
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1e-5)
    options = {"num_training_steps": 10} if kind == "cosine" else {}
    scheduler = namespace[name](optimizer, num_warmup_steps=warmup, **options)
    local = LearningRateSchedule(LRSchedule(kind), 1e-5, warmup, 10, zero_indexed_warmup=True)
    for iteration in range(10):
        assert local.lr_at(iteration) == pytest.approx(scheduler.get_last_lr()[0], abs=1e-14)
        optimizer.step()
        scheduler.step()


class _Functional:
    @staticmethod
    def masked_sum(values, mask, axis=None):  # type: ignore[no-untyped-def]
        return torch.where(mask.bool(), values, 0.0).mul(mask).sum(dim=axis)

    @classmethod
    def masked_mean(cls, values, mask, axis=None):  # type: ignore[no-untyped-def]
        return cls.masked_sum(values, mask, axis) / (mask.sum(dim=axis) + 1e-8)

    @classmethod
    def masked_var(cls, values, mask, unbiased=True):  # type: ignore[no-untyped-def]
        mean = cls.masked_mean(values, mask)
        variance = cls.masked_mean((values - mean) ** 2, mask)
        if unbiased:
            count = mask.sum()
            variance = variance * count / (count - 1)
        return variance

    @classmethod
    def masked_whiten(cls, values, mask, shift_mean=True):  # type: ignore[no-untyped-def]
        mean = cls.masked_mean(values, mask)
        output = (values - mean) * torch.rsqrt(cls.masked_var(values, mask) + 1e-8)
        return output if shift_mean else output + mean

    @staticmethod
    def clip_by_value(value, minimum, maximum):  # type: ignore[no-untyped-def]
        return torch.max(torch.min(value, maximum), minimum)


def _official():  # type: ignore[no-untyped-def]
    root = Path(os.environ.get("MINIVERL_VERL_V09_SOURCE", ".audit/upstream-verl-v0.9.0"))
    if not (root / ".git").exists():
        pytest.skip("pinned verl v0.9.0 source checkout is unavailable")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    assert commit == UPSTREAM_VERL_COMMIT
    source = root / "verl/trainer/ppo/core_algos.py"
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    names = (
        "agg_loss",
        "compute_gae_advantage_return",
        "compute_grpo_outcome_advantage",
        "compute_rloo_outcome_advantage",
        "compute_reinforce_plus_plus_outcome_advantage",
        "compute_policy_loss_vanilla",
        "compute_value_loss",
        "kl_penalty_forward",
    )
    selected = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    }
    assert set(selected) == set(names)
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            *[selected[name] for name in names],
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = {
        "torch": torch,
        "np": SimpleNamespace(ndarray=object),
        "Optional": Optional,
        "defaultdict": defaultdict,
        "verl_F": _Functional,
        "AlgoConfig": type("AlgoConfig", (), {}),
        "ActorConfig": type("ActorConfig", (), {}),
        "AdvantageEstimator": SimpleNamespace(
            GAE="gae", GRPO="grpo", RLOO="rloo", REINFORCE_PLUS_PLUS="reinforce_plus_plus"
        ),
        "register_adv_est": lambda _name: lambda fn: fn,
        "register_policy_loss": lambda _name: lambda fn: fn,
    }
    exec(compile(module, str(source), "exec"), namespace)
    return SimpleNamespace(**namespace)


class _Actor:
    clip_ratio = 0.2
    clip_ratio_low = 0.2
    clip_ratio_high = 0.2
    global_batch_info = {}  # noqa: RUF012

    def get(self, name, default):  # type: ignore[no-untyped-def]
        return getattr(self, name, default)


@pytest.mark.parametrize(
    "mode",
    [
        "token-mean",
        "token-sum",
        "seq-mean-token-sum",
        "seq-mean-token-mean",
        "seq-mean-token-sum-norm",
    ],
)
@pytest.mark.parametrize("scale", [None, 7])
def test_direct_reduction_scalar_gradient_optimizer_against_upstream(mode, scale):
    from miniverl.losses.reduction import rl_reduction_weights

    official = _official()
    old = torch.tensor([[-1.0, -0.7, -2.0], [-1.1, -0.8, -1.5]])
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    advantages = torch.tensor([[1.0, 2.0, 0.0], [-1.0, -2.0, -3.0]])
    local = torch.nn.Parameter(old.clone() + 0.15)
    upstream = torch.nn.Parameter(local.detach().clone())
    actor = _Actor()
    actor.global_batch_info = {"loss_scale_factor": scale}
    expected, metrics = official.compute_policy_loss_vanilla(
        old,
        upstream,
        advantages,
        mask,
        loss_agg_mode=mode,
        config=actor,
    )
    output = clipped_policy_loss(
        current_log_probs=local, old_log_probs=old, advantages=advantages, response_mask=mask
    )
    rows, denominator = rl_reduction_weights(
        list(mask), mode, response_length=3, loss_scale_factor=scale
    )
    actual = sum(
        (loss * weight).sum() / denominator
        for loss, weight in zip(output.per_token_loss, rows, strict=True)
    )
    expected.backward()
    actual.backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(local.grad, upstream.grad)
    assert output.metrics["pg_clipfrac"] == pytest.approx(metrics["actor/pg_clipfrac"])
    torch.optim.AdamW([local], lr=3e-4).step()
    torch.optim.AdamW([upstream], lr=3e-4).step()
    torch.testing.assert_close(local, upstream)


def test_advantage_estimators_match_pinned_upstream() -> None:
    official = _official()
    rewards = torch.tensor([[0.0, 1.0], [0.0, 3.0], [2.0, 0.0], [4.0, 0.0]])
    mask = torch.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    groups = ("a", "a", "b", "b")
    for normalize in (True, False):
        expected, expected_returns = official.compute_grpo_outcome_advantage(
            rewards.clone(), mask, groups, norm_adv_by_std_in_grpo=normalize
        )
        actual = grpo_outcome_advantage(rewards, mask, groups, normalize_by_std=normalize)
        torch.testing.assert_close(actual.advantages, expected)
        torch.testing.assert_close(actual.returns, expected_returns)
    expected, _ = official.compute_rloo_outcome_advantage(rewards.clone(), mask, groups)
    torch.testing.assert_close(rloo_outcome_advantage(rewards, mask, groups).advantages, expected)

    config = SimpleNamespace(gamma=0.9)
    expected, expected_returns = official.compute_reinforce_plus_plus_outcome_advantage(
        rewards.clone(), mask, config=config
    )
    actual = reinforce_plus_plus_advantage(rewards, mask, gamma=0.9)
    torch.testing.assert_close(actual.advantages, expected)
    torch.testing.assert_close(actual.returns, expected_returns)


def test_gae_policy_value_gradients_and_values_match_pinned_upstream() -> None:
    official = _official()
    rewards = torch.tensor([[0.0, 1.0, 0.0], [0.0, 2.0, 0.0]])
    values = torch.tensor([[0.2, 0.3, 0.0], [0.1, 0.4, 0.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    expected_adv, expected_returns = official.compute_gae_advantage_return(
        rewards, values, mask, torch.tensor(0.9), torch.tensor(0.8)
    )
    actual = gae_advantage_return(rewards, values, mask, gamma=0.9, lam=0.8)
    torch.testing.assert_close(actual.advantages, expected_adv)
    torch.testing.assert_close(actual.returns, expected_returns)

    old = torch.tensor([[-1.0, -0.7, 0.0], [-1.1, -0.8, 0.0]])
    local_current = old.clone().add(0.15).requires_grad_(True)
    upstream_current = local_current.detach().clone().requires_grad_(True)
    expected_loss, expected_metrics = official.compute_policy_loss_vanilla(
        old, upstream_current, expected_adv, mask, config=_Actor()
    )
    expected_loss.backward()
    local = clipped_policy_loss(
        current_log_probs=local_current,
        old_log_probs=old,
        advantages=actual.advantages,
        response_mask=mask,
    )
    local.loss.backward()
    torch.testing.assert_close(local.loss, expected_loss)
    torch.testing.assert_close(local_current.grad, upstream_current.grad)
    assert local.metrics["pg_clipfrac"] == pytest.approx(expected_metrics["actor/pg_clipfrac"])

    current_values = values.clone().add(0.3).requires_grad_(True)
    upstream_values = current_values.detach().clone().requires_grad_(True)
    expected_v, expected_clip = official.compute_value_loss(
        upstream_values, expected_returns, values, mask, 0.2
    )
    expected_v.backward()
    local_v = clipped_value_loss(current_values, values, actual.returns, mask, cliprange_value=0.2)
    local_v.loss.backward()
    torch.testing.assert_close(local_v.loss, expected_v)
    torch.testing.assert_close(current_values.grad, upstream_values.grad)
    assert local_v.metrics["value_clipfrac"] == pytest.approx(float(expected_clip))

    local_parameter = torch.nn.Parameter(old.clone().add(0.15))
    upstream_parameter = torch.nn.Parameter(old.clone().add(0.15))
    local_optimizer = torch.optim.AdamW([local_parameter], lr=3e-4, weight_decay=0.01)
    upstream_optimizer = torch.optim.AdamW([upstream_parameter], lr=3e-4, weight_decay=0.01)
    upstream_step_loss, _ = official.compute_policy_loss_vanilla(
        old, upstream_parameter, expected_adv, mask, config=_Actor()
    )
    upstream_step_loss.backward()
    upstream_optimizer.step()
    local_step_loss = clipped_policy_loss(
        current_log_probs=local_parameter,
        old_log_probs=old,
        advantages=actual.advantages,
        response_mask=mask,
    ).loss
    local_step_loss.backward()
    local_optimizer.step()
    torch.testing.assert_close(local_parameter, upstream_parameter)


def test_reference_kl_estimators_match_pinned_upstream() -> None:
    official = _official()
    old = torch.tensor([[-1.2, -0.3], [-3.0, -0.8]])
    reference = torch.tensor([[-1.0, -0.7], [-2.4, -1.2]])
    for local_name, upstream_name in (
        ("kl", "kl"),
        ("abs", "abs"),
        ("mse", "mse"),
        ("low_var_kl", "low_var_kl"),
    ):
        expected = official.kl_penalty_forward(old, reference, upstream_name)
        actual = reference_kl_penalty(old, reference, penalty=local_name)
        torch.testing.assert_close(actual, expected)


def test_combined_ppo_actor_and_critic_optimizer_steps_match_pinned_upstream() -> None:
    official = _official()
    mask = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    rewards = torch.tensor([[0.0, 1.0], [0.0, 2.0]])
    old_values = torch.tensor([[0.2, 0.3], [0.1, 0.4]])
    advantages, returns = official.compute_gae_advantage_return(
        rewards, old_values, mask, torch.tensor(0.9), torch.tensor(0.8)
    )
    old_log_probs = torch.tensor([[-1.0, -0.7], [-1.1, -0.8]])
    reference = old_log_probs - 0.1
    entropy = torch.tensor([[0.7, 0.8], [0.9, 1.0]])

    upstream_actor = torch.nn.Parameter(old_log_probs.clone().add(0.05))
    local_actor = torch.nn.Parameter(upstream_actor.detach().clone())
    upstream_critic = torch.nn.Parameter(old_values.clone().add(0.1))
    local_critic = torch.nn.Parameter(upstream_critic.detach().clone())
    upstream_actor_optimizer = torch.optim.AdamW([upstream_actor], lr=3e-4, weight_decay=0.01)
    local_actor_optimizer = torch.optim.AdamW([local_actor], lr=3e-4, weight_decay=0.01)
    upstream_critic_optimizer = torch.optim.AdamW([upstream_critic], lr=2e-4, weight_decay=0.02)
    local_critic_optimizer = torch.optim.AdamW([local_critic], lr=2e-4, weight_decay=0.02)

    upstream_policy, _ = official.compute_policy_loss_vanilla(
        old_log_probs, upstream_actor, advantages, mask, config=_Actor()
    )
    upstream_kl = official.kl_penalty_forward(upstream_actor, reference, "low_var_kl")
    upstream_total = (
        upstream_policy
        - 0.02 * _Functional.masked_mean(entropy, mask)
        + 0.03 * _Functional.masked_mean(upstream_kl, mask)
    )
    upstream_total.backward()
    upstream_actor_optimizer.step()

    local_policy = clipped_policy_loss(
        current_log_probs=local_actor,
        old_log_probs=old_log_probs,
        advantages=advantages,
        response_mask=mask,
    ).loss
    local_total = (
        local_policy
        - 0.02 * _Functional.masked_mean(entropy, mask)
        + 0.03
        * _Functional.masked_mean(
            reference_kl_penalty(local_actor, reference, penalty="low_var_kl"), mask
        )
    )
    local_total.backward()
    local_actor_optimizer.step()

    upstream_value_loss, _ = official.compute_value_loss(
        upstream_critic, returns, old_values, mask, 0.2
    )
    upstream_value_loss.backward()
    upstream_critic_optimizer.step()
    local_value_loss = clipped_value_loss(
        local_critic, old_values, returns, mask, cliprange_value=0.2
    ).loss
    local_value_loss.backward()
    local_critic_optimizer.step()

    torch.testing.assert_close(local_total, upstream_total)
    torch.testing.assert_close(local_value_loss, upstream_value_loss)
    torch.testing.assert_close(local_actor, upstream_actor)
    torch.testing.assert_close(local_critic, upstream_critic)
