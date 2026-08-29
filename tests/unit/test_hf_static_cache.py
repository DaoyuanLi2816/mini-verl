# ruff: noqa: E402

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.torch

from miniverl.models import hf


def test_static_cache_factory_supports_transformers_4_signature(monkeypatch) -> None:
    observed = {}

    class LegacyStaticCache:
        def __init__(
            self,
            config,
            max_batch_size,
            max_cache_len,
            device,
            dtype,
        ) -> None:
            observed.update(locals())

    monkeypatch.setattr(
        hf,
        "require_transformers",
        lambda _feature: SimpleNamespace(StaticCache=LegacyStaticCache),
    )
    config = object()

    hf._static_generation_cache(
        config,
        batch_size=4,
        max_cache_len=768,
        device="cuda",
        dtype=torch.bfloat16,
    )

    assert observed["config"] is config
    assert observed["max_batch_size"] == 4
    assert observed["max_cache_len"] == 768
    assert observed["device"] == "cuda"
    assert observed["dtype"] is torch.bfloat16


def test_static_cache_factory_supports_transformers_5_signature(monkeypatch) -> None:
    observed = {}

    class ModernStaticCache:
        def __init__(self, config, max_cache_len, offloading=False, **kwargs) -> None:
            observed.update(locals())

    monkeypatch.setattr(
        hf,
        "require_transformers",
        lambda _feature: SimpleNamespace(StaticCache=ModernStaticCache),
    )
    config = object()

    hf._static_generation_cache(
        config,
        batch_size=4,
        max_cache_len=768,
        device="cuda",
        dtype=torch.bfloat16,
    )

    assert observed["config"] is config
    assert observed["max_cache_len"] == 768
    assert observed["offloading"] is False
    assert observed["kwargs"] == {}
