"""Composable verl-conformant single-GPU RL mathematical primitives.

The implementations use :mod:`torch`, but importing this package remains safe
for the dependency-light core install. Individual symbols are resolved only
when used, following the same contract as :mod:`miniverl.losses`.
"""

from __future__ import annotations

from typing import Any

from miniverl.errors import MissingDependencyError
from miniverl.utils.lazy import have_module

__all__ = [
    "AdvantageOutput",
    "KLPenalty",
    "PolicyLossOutput",
    "ValueLossOutput",
    "clipped_policy_loss",
    "clipped_value_loss",
    "gae_advantage_return",
    "grpo_outcome_advantage",
    "reinforce_plus_plus_advantage",
    "reference_kl_penalty",
    "rloo_outcome_advantage",
]

_MODULE_OF = {
    "AdvantageOutput": "advantages",
    "gae_advantage_return": "advantages",
    "grpo_outcome_advantage": "advantages",
    "reinforce_plus_plus_advantage": "advantages",
    "rloo_outcome_advantage": "advantages",
    "KLPenalty": "kl",
    "reference_kl_penalty": "kl",
    "PolicyLossOutput": "policy",
    "clipped_policy_loss": "policy",
    "ValueLossOutput": "value",
    "clipped_value_loss": "value",
}


def __getattr__(name: str) -> Any:
    submodule = _MODULE_OF.get(name)
    if submodule is None:
        raise AttributeError(f"module 'miniverl.algorithms' has no attribute {name!r}")
    if not have_module("torch"):
        raise MissingDependencyError("torch", "train", f"miniverl.algorithms.{name}")
    import importlib

    module = importlib.import_module(f"miniverl.algorithms.{submodule}")
    return getattr(module, name)


def __dir__() -> list[str]:
    return sorted(__all__)
