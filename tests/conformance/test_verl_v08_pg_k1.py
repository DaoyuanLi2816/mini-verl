# ruff: noqa: E402 - verify the optional official checkout before torch-backed imports

"""Scalar, metric, gradient and optimizer-step conformance for pinned PG-k1."""

from __future__ import annotations

import ast
import importlib.metadata
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar
from urllib.parse import unquote, urlparse

import pytest

torch = pytest.importorskip("torch")

from miniverl.bridge.opd_v08 import VERL_COMMIT
from miniverl.losses.verl_pg import verl_pg_k1_loss

pytestmark = [pytest.mark.torch, pytest.mark.verl_conformance]


class _VerlFunctional:
    @staticmethod
    def masked_sum(value, mask):  # type: ignore[no-untyped-def]
        return (value * mask).sum()

    @staticmethod
    def masked_mean(value, mask):  # type: ignore[no-untyped-def]
        return (value * mask).sum() / mask.sum()


def _official_functions():  # type: ignore[no-untyped-def]
    try:
        distribution = importlib.metadata.distribution("verl")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("official verl v0.8.0 is not installed")
    direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
    vcs_info = direct_url.get("vcs_info")
    if vcs_info is not None:
        assert vcs_info["commit_id"] == VERL_COMMIT
    else:
        checkout = Path(unquote(urlparse(direct_url["url"]).path.lstrip("/")))
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=checkout,
            check=True,
            capture_output=True,
            text=True,
        )
        assert completed.stdout.strip() == VERL_COMMIT
    source = Path(distribution.locate_file("")) / "verl" / "trainer" / "ppo" / "core_algos.py"
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    selected = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in {"agg_loss", "compute_policy_loss_vanilla", "kl_penalty_forward"}
    }
    assert set(selected) == {"agg_loss", "compute_policy_loss_vanilla", "kl_penalty_forward"}
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            selected["agg_loss"],
            selected["compute_policy_loss_vanilla"],
            selected["kl_penalty_forward"],
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = {
        "torch": torch,
        "verl_F": _VerlFunctional,
        "AlgoConfig": type("AlgoConfig", (), {}),
        "register_policy_loss": lambda _name: lambda function: function,
    }
    exec(compile(module, str(source), "exec"), namespace)
    return SimpleNamespace(**namespace)


class _PolicyConfig:
    clip_ratio = 0.2
    clip_ratio_low = 0.2
    clip_ratio_high = 0.2
    global_batch_info: ClassVar[dict[str, object]] = {}

    def get(self, name: str, default):  # type: ignore[no-untyped-def]
        return getattr(self, name, default)


def test_pg_k1_scalar_metrics_gradient_and_step_match_exact_pinned_functions() -> None:
    official = _official_functions()
    logits_data = torch.tensor(
        [[0.4, -0.2, 0.1, 0.8], [-0.3, 1.0, 0.2, -0.1], [0.7, 0.0, -0.4, 0.3]],
        dtype=torch.float32,
    )
    sampled = torch.tensor([3, 1, 0])
    old = torch.tensor([-0.9, -0.7, -1.4])
    teacher = torch.tensor([-0.4, -1.1, -0.8])
    mask = torch.tensor([1.0, 1.0, 1.0])

    official_logits = logits_data.clone().requires_grad_(True)
    official_current = (
        torch.log_softmax(official_logits, dim=-1).gather(-1, sampled[:, None]).squeeze(-1)
    )
    estimator = official.kl_penalty_forward(old, teacher, "k1")
    official_loss, official_metrics = official.compute_policy_loss_vanilla(
        old_log_prob=old,
        log_prob=official_current,
        advantages=-estimator.detach(),
        response_mask=mask,
        loss_agg_mode="token-mean",
        config=_PolicyConfig(),
    )
    official_loss.backward()

    local_logits = logits_data.clone().requires_grad_(True)
    local_current = torch.log_softmax(local_logits, dim=-1).gather(-1, sampled[:, None]).squeeze(-1)
    local = verl_pg_k1_loss(
        current_log_probs=local_current,
        old_log_probs=old,
        teacher_log_probs=teacher,
        weights=mask,
    )
    local.loss.backward()

    torch.testing.assert_close(local.estimator, estimator, rtol=0.0, atol=0.0)
    torch.testing.assert_close(local.loss, official_loss, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(local_logits.grad, official_logits.grad, rtol=1e-6, atol=1e-7)
    assert local.metrics["pg_clipfrac"] == pytest.approx(official_metrics["actor/pg_clipfrac"])
    assert local.metrics["ppo_kl"] == pytest.approx(official_metrics["actor/ppo_kl"])
    assert local.metrics["pg_clipfrac_lower"] == pytest.approx(
        official_metrics["actor/pg_clipfrac_lower"]
    )

    official_parameter = torch.nn.Parameter(logits_data.clone())
    local_parameter = torch.nn.Parameter(logits_data.clone())
    official_optimizer = torch.optim.SGD([official_parameter], lr=0.03)
    local_optimizer = torch.optim.SGD([local_parameter], lr=0.03)
    official_current = (
        torch.log_softmax(official_parameter, dim=-1).gather(-1, sampled[:, None]).squeeze(-1)
    )
    official_step_loss, _ = official.compute_policy_loss_vanilla(
        old_log_prob=old,
        log_prob=official_current,
        advantages=-estimator.detach(),
        response_mask=mask,
        loss_agg_mode="token-mean",
        config=_PolicyConfig(),
    )
    official_step_loss.backward()
    official_optimizer.step()
    local_current = (
        torch.log_softmax(local_parameter, dim=-1).gather(-1, sampled[:, None]).squeeze(-1)
    )
    local_step = verl_pg_k1_loss(
        current_log_probs=local_current,
        old_log_probs=old,
        teacher_log_probs=teacher,
        weights=mask,
    )
    local_step.loss.backward()
    local_optimizer.step()
    torch.testing.assert_close(local_parameter, official_parameter, rtol=1e-6, atol=1e-7)


def test_teacher_logprob_perturbation_reverses_sampled_action_gradient() -> None:
    logits = torch.tensor([[0.1, 0.2, -0.3]], requires_grad=True)
    sampled = torch.tensor([1])
    current = torch.log_softmax(logits, dim=-1).gather(-1, sampled[:, None]).squeeze(-1)
    old = current.detach()
    favored = verl_pg_k1_loss(
        current_log_probs=current,
        old_log_probs=old,
        teacher_log_probs=old + 0.75,
        weights=torch.ones(1),
    )
    favored.loss.backward()
    favored_gradient = logits.grad.detach().clone()
    logits.grad = None
    current = torch.log_softmax(logits, dim=-1).gather(-1, sampled[:, None]).squeeze(-1)
    disfavored = verl_pg_k1_loss(
        current_log_probs=current,
        old_log_probs=old,
        teacher_log_probs=old - 0.75,
        weights=torch.ones(1),
    )
    disfavored.loss.backward()
    assert logits.grad is not None
    assert favored_gradient[0, 1] < 0 < logits.grad[0, 1]
