"""Small, dependency-light helpers shared across miniVERL."""

from __future__ import annotations

from miniverl.utils.lazy import (
    have_module,
    require_bitsandbytes,
    require_module,
    require_peft,
    require_torch,
    require_transformers,
)

__all__ = [
    "have_module",
    "require_module",
    "require_torch",
    "require_transformers",
    "require_peft",
    "require_bitsandbytes",
]
