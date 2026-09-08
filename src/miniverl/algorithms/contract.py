"""Torch-free identity for the pinned verl RL semantics."""

from __future__ import annotations

UPSTREAM_VERL_TAG = "v0.9.0"
UPSTREAM_VERL_COMMIT = "483b8a009ba3a97563edee3a19887e4862b8094a"
ADVANTAGE_IMPLEMENTATION_VERSION = "verl-v0.9-advantages-v1"

__all__ = [
    "ADVANTAGE_IMPLEMENTATION_VERSION",
    "UPSTREAM_VERL_COMMIT",
    "UPSTREAM_VERL_TAG",
]
