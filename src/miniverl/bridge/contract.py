"""Immutable compatibility identity for the v0.6 verl bridge."""

from __future__ import annotations

from typing import Final

from miniverl.errors import ConfigError

__all__ = [
    "BRIDGE_PROFILE",
    "COMPATIBILITY_LEVELS",
    "TESTED_PYTHON",
    "VERL_COMMIT",
    "VERL_REPOSITORY",
    "VERL_TAG",
    "required_verl_text",
    "validate_target_verl",
]

VERL_REPOSITORY: Final = "https://github.com/verl-project/verl"
VERL_TAG: Final = "v0.8.0"
VERL_COMMIT: Final = "7aed6b230776f963fa09509c10d9c3a767d1102c"
TESTED_PYTHON: Final = "3.12"
BRIDGE_PROFILE: Final = "single-gpu-online-distillation-v1"

COMPATIBILITY_LEVELS: Final = {
    0: "conceptual post-training flow",
    1: "standard artifact interoperability",
    2: "versioned config-field whitelist",
    3: "miniVERL-defined validated pinned artifact bundle",
}


def validate_target_verl(value: str) -> str:
    """Resolve the only accepted tag/commit and reject moving or guessed targets."""
    normalized = value.strip()
    if normalized in {VERL_TAG, VERL_COMMIT}:
        return VERL_TAG
    raise ConfigError(
        f"unsupported or moving verl target {value!r}; expected the pinned verl target "
        f"{VERL_TAG!r} or commit {VERL_COMMIT}",
        hint="use --target-verl v0.8.0 for the released bridge profile",
    )


def required_verl_text() -> str:
    """Machine-readable pin placed in every exported recipe."""
    return (
        f"VERL_REPOSITORY={VERL_REPOSITORY}\n"
        f"VERL_TAG={VERL_TAG}\n"
        f"VERL_COMMIT={VERL_COMMIT}\n"
        f"TESTED_PYTHON={TESTED_PYTHON}\n"
        f"PROFILE={BRIDGE_PROFILE}\n"
        "OBSERVED_PACKAGE_VERSION=0.8.0.dev0\n"
    )
