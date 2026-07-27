"""Teacher scoring: turning student states into token-level supervision.

The contract in :mod:`miniverl.teachers.base` is pure Python, but the only
bundled implementation runs a model, so
:class:`~miniverl.teachers.local.LocalTeacherScorer` needs :mod:`torch`.  It is
therefore resolved through the module-level ``__getattr__`` below -- the same
idiom :mod:`miniverl.losses` uses -- which keeps ``import miniverl.teachers``
working without the ``train`` extra and turns the eventual failure into an
actionable :class:`~miniverl.errors.MissingDependencyError` rather than a bare
``ModuleNotFoundError`` naming a package the reader never asked for.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from miniverl.errors import MissingDependencyError
from miniverl.teachers.base import TeacherScorer, TeacherScoreResult
from miniverl.utils.lazy import have_module

if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    from miniverl.teachers.local import LocalTeacherScorer

__all__ = ["TeacherScorer", "TeacherScoreResult", "LocalTeacherScorer"]

#: Names that cost a torch import, mapped to the submodule that defines them.
_MODULE_OF = {"LocalTeacherScorer": "local"}


def __getattr__(name: str) -> Any:
    submodule = _MODULE_OF.get(name)
    if submodule is None:
        raise AttributeError(f"module 'miniverl.teachers' has no attribute {name!r}")
    if not have_module("torch"):
        raise MissingDependencyError("torch", "train", f"miniverl.teachers.{name}")
    import importlib

    module = importlib.import_module(f"miniverl.teachers.{submodule}")
    return getattr(module, name)


def __dir__() -> list[str]:
    return sorted(__all__)
