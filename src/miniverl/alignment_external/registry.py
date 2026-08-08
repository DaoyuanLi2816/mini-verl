"""Load and validate the pinned external endpoint registry.

The registry is the contract: it says which dataset revision, which evaluator
revision and which license each endpoint runs under. Loading it is where a
drifted pin is caught, so the validation is strict and every failure is loud.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "REQUIRED_CATEGORIES",
    "Endpoint",
    "load_registry",
    "validate_registry",
]

REQUIRED_CATEGORIES = (
    "instruction_following",
    "over_refusal",
    "harmful_compliance",
    "preference_reward",
)

_REQUIRED_ENDPOINT_FIELDS = (
    "id",
    "category",
    "name",
    "dataset",
    "revision",
    "split",
    "license",
    "gated",
    "prompt_redistribution",
    "output_redistribution",
    "evaluator",
    "known_limitations",
)

_HEX40 = 40


class Endpoint(dict):
    """One registry entry. A dict so it serialises straight into reports."""

    @property
    def id(self) -> str:
        return str(self["id"])

    @property
    def evaluator_model(self) -> str | None:
        model = self["evaluator"].get("model")
        return str(model) if model else None

    @property
    def requires_qualification(self) -> bool:
        return bool(self["evaluator"].get("requires_qualification", False))


def default_registry_path(root: Path | None = None) -> Path:
    base = root or Path(__file__).resolve().parents[3]
    return base / "benchmarks" / "external-alignment" / "registry.yaml"


def load_registry(path: str | Path | None = None) -> dict[str, Any]:
    """Read the registry and fail closed on any structural problem."""
    target = Path(path) if path is not None else default_registry_path()
    try:
        payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot read endpoint registry {target}: {exc}") from exc
    problems = validate_registry(payload)
    if problems:
        listing = "\n".join(f"  - {problem}" for problem in problems)
        raise ValueError(f"endpoint registry {target} is invalid:\n{listing}")
    payload["endpoints"] = [Endpoint(entry) for entry in payload["endpoints"]]
    return payload


def validate_registry(payload: Any) -> list[str]:
    """Every structural problem in the registry, empty when it is sound."""
    problems: list[str] = []
    if not isinstance(payload, dict):
        return ["the registry must be a YAML mapping"]
    if payload.get("schema_version") != 1:
        problems.append(f"schema_version {payload.get('schema_version')!r} is not 1")

    endpoints = payload.get("endpoints")
    if not isinstance(endpoints, list) or not endpoints:
        return [*problems, "endpoints must be a non-empty list"]

    seen_ids: set[str] = set()
    covered: set[str] = set()
    for index, entry in enumerate(endpoints):
        if not isinstance(entry, dict):
            problems.append(f"endpoint {index} is not a mapping")
            continue
        name = entry.get("id", f"#{index}")
        for required in _REQUIRED_ENDPOINT_FIELDS:
            if entry.get(required) in (None, ""):
                problems.append(f"endpoint {name}: missing {required}")
        if entry.get("id") in seen_ids:
            problems.append(f"endpoint {name}: duplicate id")
        seen_ids.add(str(entry.get("id")))
        covered.add(str(entry.get("category")))

        revision = str(entry.get("revision", ""))
        if len(revision) != _HEX40 or not all(c in "0123456789abcdef" for c in revision):
            problems.append(
                f"endpoint {name}: revision must be a 40-character hex commit, got {revision!r}"
            )
        # A gated source cannot be reproduced by a reader, so it cannot be a
        # pinned endpoint however convenient it is.
        if entry.get("gated") not in (False, None):
            problems.append(
                f"endpoint {name}: gated sources are not reproducible by a reader "
                f"(gated={entry.get('gated')!r})"
            )

        evaluator = entry.get("evaluator")
        if not isinstance(evaluator, dict):
            problems.append(f"endpoint {name}: evaluator must be a mapping")
            continue
        if evaluator.get("kind") not in {
            "deterministic_rule",
            "classifier_model",
            "pairwise_model",
        }:
            problems.append(f"endpoint {name}: unknown evaluator kind {evaluator.get('kind')!r}")
        if evaluator.get("model"):
            model_revision = str(evaluator.get("model_revision", ""))
            if len(model_revision) != _HEX40:
                problems.append(
                    f"endpoint {name}: evaluator model needs a pinned 40-character revision"
                )
            if evaluator.get("model_gated") not in (False, None):
                problems.append(f"endpoint {name}: evaluator model is gated and not reproducible")
            parameters = evaluator.get("model_parameters_b")
            if not isinstance(parameters, (int, float)) or parameters > 3.0:
                problems.append(
                    f"endpoint {name}: evaluator model is {parameters}B; the compute "
                    "contract caps a judge at 3B"
                )
            if not evaluator.get("requires_qualification"):
                problems.append(f"endpoint {name}: a model evaluator must be qualified before use")

    missing = [category for category in REQUIRED_CATEGORIES if category not in covered]
    if missing:
        problems.append(f"no endpoint covers required categor(ies): {', '.join(missing)}")
    return problems
