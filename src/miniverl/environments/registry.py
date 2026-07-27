"""Environment registry.

Registration is explicit -- there is no plugin discovery, no entry-point scan
and no import of arbitrary user modules by name.  Adding an environment means
importing it here, which is also what makes the set of things a recipe can
execute auditable.
"""

from __future__ import annotations

from typing import Any

from miniverl.environments.base import ToolEnvironment
from miniverl.environments.calculator import CalculatorEnvironment
from miniverl.environments.jsonnav import JsonNavEnvironment
from miniverl.environments.sqlite_env import SqliteEnvironment
from miniverl.errors import ConfigError

__all__ = ["make_environment", "available_environments", "ENVIRONMENT_NAMES", "register"]

_REGISTRY: dict[str, type[ToolEnvironment]] = {
    CalculatorEnvironment.name: CalculatorEnvironment,
    JsonNavEnvironment.name: JsonNavEnvironment,
    SqliteEnvironment.name: SqliteEnvironment,
}

#: Names accepted by ``environment.name`` in a recipe.
ENVIRONMENT_NAMES: tuple[str, ...] = tuple(sorted(_REGISTRY))


def register(cls: type[ToolEnvironment]) -> type[ToolEnvironment]:
    """Register a custom environment class (used by the examples)."""
    if not cls.name or cls.name == "base":
        raise ConfigError("an environment class must set a unique 'name' attribute")
    _REGISTRY[cls.name] = cls
    return cls


def available_environments() -> list[str]:
    """Sorted list of registered environment names."""
    return sorted(_REGISTRY)


def make_environment(name: str, **params: Any) -> ToolEnvironment:
    """Instantiate a registered environment."""
    cls = _REGISTRY.get(name)
    if cls is None:
        raise ConfigError(
            f"unknown environment {name!r}",
            hint=f"available environments: {', '.join(available_environments())}",
        )
    return cls(**params)
