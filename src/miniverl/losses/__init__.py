"""Divergence objectives.

Importing a name from this package pulls in :mod:`torch`.  The module-level
``__getattr__`` below turns the resulting ``ModuleNotFoundError`` into an
actionable :class:`~miniverl.errors.MissingDependencyError`, so
``import miniverl.losses`` itself stays free and only *use* costs a torch
import.
"""

from __future__ import annotations

from typing import Any

from miniverl.errors import MissingDependencyError
from miniverl.utils.lazy import have_module

__all__ = [
    # numerics
    "log1mexp",
    "kl_from_log_probs",
    "entropy_from_log_probs",
    "log_softmax_f32",
    # exact
    "exact_forward_kl",
    "exact_reverse_kl",
    "exact_jsd",
    "exact_divergence",
    "exact_teacher_entropy",
    # bucketed
    "teacher_topk_targets",
    "student_bucket_log_probs",
    "build_bucket_distributions",
    "bucketed_forward_kl",
    "bucketed_reverse_kl",
    "bucketed_jsd",
    "bucketed_divergence",
    "bucketed_teacher_entropy",
    # reduction / chunking
    "weighted_mean",
    "total_weight",
    "LossOutput",
    "ExactTargetProvider",
    "BucketedTargetProvider",
    "chunked_selected_position_loss",
]

_MODULE_OF: dict[str, str] = {
    "log1mexp": "numerics",
    "kl_from_log_probs": "numerics",
    "entropy_from_log_probs": "numerics",
    "log_softmax_f32": "numerics",
    "exact_forward_kl": "exact",
    "exact_reverse_kl": "exact",
    "exact_jsd": "exact",
    "exact_divergence": "exact",
    "exact_teacher_entropy": "exact",
    "teacher_topk_targets": "bucketed",
    "student_bucket_log_probs": "bucketed",
    "build_bucket_distributions": "bucketed",
    "bucketed_forward_kl": "bucketed",
    "bucketed_reverse_kl": "bucketed",
    "bucketed_jsd": "bucketed",
    "bucketed_divergence": "bucketed",
    "bucketed_teacher_entropy": "bucketed",
    "weighted_mean": "reduction",
    "total_weight": "reduction",
    "LossOutput": "chunked",
    "ExactTargetProvider": "chunked",
    "BucketedTargetProvider": "chunked",
    "chunked_selected_position_loss": "chunked",
}


def __getattr__(name: str) -> Any:
    submodule = _MODULE_OF.get(name)
    if submodule is None:
        raise AttributeError(f"module 'miniverl.losses' has no attribute {name!r}")
    if not have_module("torch"):
        raise MissingDependencyError("torch", "train", f"miniverl.losses.{name}")
    import importlib

    module = importlib.import_module(f"miniverl.losses.{submodule}")
    return getattr(module, name)


def __dir__() -> list[str]:
    return sorted(__all__)
