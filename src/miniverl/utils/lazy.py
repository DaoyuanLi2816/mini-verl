"""Lazy imports for optional heavy dependencies.

miniVERL's core is importable without torch.  Every heavy import goes through
:func:`require_module` so a missing dependency produces an actionable
:class:`~miniverl.errors.MissingDependencyError` instead of a bare
``ModuleNotFoundError``.
"""

from __future__ import annotations

import importlib
import importlib.util
from types import ModuleType
from typing import Any

from miniverl.errors import MissingDependencyError

__all__ = [
    "have_module",
    "require_module",
    "require_torch",
    "require_transformers",
    "require_peft",
    "require_bitsandbytes",
]


def have_module(name: str) -> bool:
    """Return ``True`` if ``name`` can be imported without importing it."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):  # pragma: no cover - defensive
        return False


def require_module(name: str, extra: str, purpose: str) -> ModuleType:
    """Import ``name`` or raise an actionable :class:`MissingDependencyError`.

    Parameters
    ----------
    name:
        Importable module name, for example ``"torch"``.
    extra:
        The miniVERL extra that provides it, for example ``"train"``.
    purpose:
        Human-readable description of what needed it, used in the message.
    """
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise MissingDependencyError(name, extra, purpose) from exc


def require_torch(purpose: str = "This operation") -> Any:
    """Import and return :mod:`torch`."""
    return require_module("torch", "train", purpose)


def require_transformers(purpose: str = "This operation") -> Any:
    """Import and return :mod:`transformers`."""
    return require_module("transformers", "train", purpose)


def require_peft(purpose: str = "This operation") -> Any:
    """Import and return :mod:`peft`."""
    return require_module("peft", "train", purpose)


def require_bitsandbytes(purpose: str = "4-bit quantization") -> Any:
    """Import and return :mod:`bitsandbytes`."""
    return require_module("bitsandbytes", "cuda", purpose)
