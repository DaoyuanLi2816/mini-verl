"""Shared pytest fixtures and collection rules.

Tests are split by dependency weight:

* ``tests/unit`` and ``tests/property`` -- pure Python, no torch import
  (except the numerical modules, which are marked ``torch``).
* ``tests/integration`` -- CPU torch.
* ``tests/gpu`` -- marked ``gpu`` and deselected unless CUDA is present.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Iterator
from pathlib import Path

import pytest

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None
HAS_PEFT = importlib.util.find_spec("peft") is not None


def _cuda_available() -> bool:
    if not HAS_TORCH:
        return False
    import torch

    return bool(torch.cuda.is_available())


HAS_CUDA = _cuda_available()

requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="requires the [train] extra (torch)")
requires_transformers = pytest.mark.skipif(
    not (HAS_TORCH and HAS_TRANSFORMERS), reason="requires torch + transformers"
)
requires_peft = pytest.mark.skipif(
    not (HAS_TORCH and HAS_TRANSFORMERS and HAS_PEFT), reason="requires torch + transformers + peft"
)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Auto-skip GPU tests when there is no CUDA device."""
    if HAS_CUDA:
        return
    skip_gpu = pytest.mark.skip(reason="no CUDA device available")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)


@pytest.fixture
def run_dir(tmp_path: Path) -> Iterator[Path]:
    """An isolated output directory for a run."""
    target = tmp_path / "runs"
    target.mkdir(parents=True, exist_ok=True)
    yield target


@pytest.fixture
def toy_tokenizer():  # type: ignore[no-untyped-def]
    """A fresh toy tokenizer."""
    from miniverl.models.tokenizers import ToyTokenizer

    return ToyTokenizer()
