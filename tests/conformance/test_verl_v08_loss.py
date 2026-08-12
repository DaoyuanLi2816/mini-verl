"""Numerical conformance against the exact installed official verl source."""

from __future__ import annotations

# ruff: noqa: E402 - skip collection before importing torch-backed miniVERL code

import importlib.metadata
import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import unquote, urlparse

import pytest

torch = pytest.importorskip("torch")

from miniverl.bridge.opd_v08 import VERL_COMMIT
from miniverl.losses.verl_topk import verl_forward_kl_topk

pytestmark = [pytest.mark.torch, pytest.mark.verl_conformance]


def _official_module():  # type: ignore[no-untyped-def]
    try:
        distribution = importlib.metadata.distribution("verl")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("official verl v0.8.0 is not installed")
    direct_url_text = distribution.read_text("direct_url.json")
    assert direct_url_text is not None, "official verl install has no VCS provenance"
    direct_url = json.loads(direct_url_text)
    vcs_info = direct_url.get("vcs_info")
    if vcs_info is not None:
        assert vcs_info["commit_id"] == VERL_COMMIT
    else:
        parsed = urlparse(direct_url["url"])
        checkout = Path(unquote(parsed.path.lstrip("/")))
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=checkout,
            check=True,
            capture_output=True,
            text=True,
        )
        assert completed.stdout.strip() == VERL_COMMIT
    root = Path(distribution.locate_file(""))
    source = root / "verl" / "trainer" / "distillation" / "fsdp" / "losses.py"
    assert source.is_file(), source

    # Load the official file directly so the conformance environment does not
    # need Ray or the distributed worker stack imported by verl.__init__.
    ulysses = types.ModuleType("verl.utils.ulysses")
    ulysses.get_ulysses_sequence_parallel_world_size = lambda: 1
    ulysses.slice_input_tensor = lambda value, dim: value
    config = types.ModuleType("verl.workers.config")
    config.DistillationConfig = type("DistillationConfig", (), {})
    config.DistillationLossConfig = type("DistillationLossConfig", (), {})
    saved = {
        name: sys.modules.get(name)
        for name in (
            "verl",
            "verl.utils",
            "verl.utils.ulysses",
            "verl.workers",
            "verl.workers.config",
        )
    }
    sys.modules["verl"] = types.ModuleType("verl")
    sys.modules["verl.utils"] = types.ModuleType("verl.utils")
    sys.modules["verl.utils.ulysses"] = ulysses
    sys.modules["verl.workers"] = types.ModuleType("verl.workers")
    sys.modules["verl.workers.config"] = config
    try:
        spec = importlib.util.spec_from_file_location("_official_verl_v08_fsdp_losses", source)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def _nested(rows: torch.Tensor) -> torch.Tensor:
    offsets = torch.tensor([0, rows.shape[0]], dtype=torch.int64)
    return torch.nested.nested_tensor_from_jagged(rows, offsets=offsets)


@pytest.mark.parametrize("minimum", [None, -10.0])
def test_values_diagnostics_reduction_and_gradient_match_official_verl(minimum) -> None:
    official = _official_module()
    logits_data = torch.tensor(
        [
            [0.2, -0.4, 1.3, 0.1, 0.7],
            [-0.8, 1.1, 0.4, 0.2, 0.9],
            [1.4, 0.3, -0.6, 0.8, 0.0],
        ],
        dtype=torch.float32,
    )
    teacher_ids = torch.tensor([[2, 4], [1, 4], [0, 3]], dtype=torch.int64)
    teacher_log_probs = torch.log(
        torch.tensor([[0.62, 0.21], [0.58, 0.25], [0.54, 0.30]], dtype=torch.float32)
    )

    official_logits = logits_data.clone().unsqueeze(0).requires_grad_(True)
    official_output = official.compute_forward_kl_topk(
        student_logits=official_logits,
        teacher_topk_log_probs=_nested(teacher_log_probs),
        teacher_topk_ids=_nested(teacher_ids),
        config=SimpleNamespace(distillation_loss=SimpleNamespace(log_prob_min_clamp=minimum)),
        data_format="thd",
    )
    official_loss = official_output["distillation_losses"].clamp_min(0.0)
    official_scalar = official_loss.mean()
    official_scalar.backward()

    local_logits = logits_data.clone().requires_grad_(True)
    local = verl_forward_kl_topk(
        local_logits,
        teacher_log_probs,
        teacher_ids,
        log_prob_min_clamp=minimum,
    )
    local_scalar = local.loss.mean()
    local_scalar.backward()

    torch.testing.assert_close(local.loss, official_loss.squeeze(0), rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(local.student_mass, official_output["student_mass"].squeeze(0))
    torch.testing.assert_close(local.teacher_mass, official_output["teacher_mass"].squeeze(0))
    torch.testing.assert_close(local.overlap_count, official_output["overlap_count"].squeeze(0))
    torch.testing.assert_close(
        local.overlap_token_advantage,
        official_output["overlap_token_advantage"].squeeze(0),
    )
    torch.testing.assert_close(local_scalar, official_scalar)
    torch.testing.assert_close(
        local_logits.grad, official_logits.grad.squeeze(0), rtol=1e-6, atol=1e-7
    )
