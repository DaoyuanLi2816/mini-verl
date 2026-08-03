"""Pinned external benchmark adapter registry."""

from __future__ import annotations

from importlib.resources import files

import yaml

from miniverl.alignment.schema import BenchmarkRegistry

__all__ = ["load_benchmark_registry"]


def load_benchmark_registry() -> BenchmarkRegistry:
    """Load the package-bundled, revision-pinned metadata registry."""
    text = (
        files("miniverl.alignment").joinpath("benchmark_registry.yaml").read_text(encoding="utf-8")
    )
    return BenchmarkRegistry.model_validate(yaml.safe_load(text))
