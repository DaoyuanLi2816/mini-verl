"""One shared, conservative audit for unresolved Hydra/OmegaConf interpolation.

miniVERL never resolves ``${...}`` on the user's behalf. Resolution would need
the original Hydra search path, the original working directory and -- for
``${oc.env:...}`` and ``${env:...}`` -- the original process environment, none
of which are part of a pinned, reviewable bridge input. An unresolved token is
therefore a hard input defect, not something to guess at.

Detection is deliberately conservative: any ``${`` in a reachable string is a
finding, including an unterminated ``"${a"`` and an escaped ``"\\${a}"``. A
false positive costs the user one explicit edit; a false negative puts a
literal ``${MODEL_PATH}`` into a recipe that miniVERL reported as validated.

Only the standard library is used, so the torch-free core keeps its dependency
surface.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from miniverl.errors import ConfigError

__all__ = [
    "MARKER",
    "audit_interpolation",
    "contains_interpolation",
    "first_token",
    "reject_interpolation",
]

MARKER = "${"
_TOKEN = re.compile(r"\$\{[^{}]*\}")


def first_token(value: str) -> str:
    """Return the first complete ``${...}`` token, or the bare marker.

    An unterminated or nested form has no complete token to quote, and the
    marker itself is what the user needs to find in their source.
    """
    match = _TOKEN.search(value)
    return match.group(0) if match else MARKER


def contains_interpolation(value: Any) -> bool:
    """Whether ``value`` reaches any string carrying an interpolation marker.

    Deliberately defined in terms of :func:`audit_interpolation` so the cheap
    predicate and the reporting walk can never disagree about what counts.
    """
    return bool(audit_interpolation(value))


def audit_interpolation(value: Any, *, label: str = "value") -> list[dict[str, str]]:
    """Walk ``value`` and return every unresolved interpolation finding.

    Strings, mappings, lists and tuples are traversed. Each finding carries the
    dotted/indexed ``location`` inside ``label`` and the offending ``token``.
    """
    findings: list[dict[str, str]] = []
    _walk(value, label, findings)
    return findings


def reject_interpolation(value: Any, *, label: str, hint: str | None = None) -> None:
    """Raise :class:`ConfigError` if ``value`` carries any unresolved token."""
    findings = audit_interpolation(value, label=label)
    if not findings:
        return
    detail = "; ".join(f"{item['location']} = {item['token']}" for item in findings[:5])
    if len(findings) > 5:
        detail += f"; and {len(findings) - 5} more"
    raise ConfigError(
        f"{label} contains unresolved interpolation: {detail}",
        hint=(
            hint
            or "resolve every ${...} value explicitly before importing; miniVERL never "
            "expands environment variables or Hydra references on your behalf"
        ),
    )


def _walk(value: Any, location: str, findings: list[dict[str, str]]) -> None:
    if isinstance(value, str):
        if MARKER in value:
            findings.append({"location": location, "token": first_token(value)})
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{location}.{key}"
            if isinstance(key, str) and MARKER in key:
                findings.append({"location": child, "token": first_token(key)})
            _walk(item, child, findings)
        return
    # ``str`` and ``bytes`` are Sequences; both are handled or ignored above.
    if isinstance(value, (list, tuple)) or (
        isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))
    ):
        for index, item in enumerate(value):
            _walk(item, f"{location}[{index}]", findings)
        return
    if isinstance(value, (set, frozenset)):
        # Sets carry no source order; sort by repr so findings stay deterministic.
        for index, item in enumerate(sorted(value, key=repr)):
            _walk(item, f"{location}{{{index}}}", findings)
