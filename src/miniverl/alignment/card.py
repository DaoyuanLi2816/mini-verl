"""Privacy-safe, hash-bound Alignment Card publication."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from miniverl.alignment.schema import AlignmentMethod, AlignmentMetrics
from miniverl.utils.runs import canonical_json, write_json_atomic, write_text

__all__ = ["render_alignment_card"]


def render_alignment_card(
    destination: str | Path,
    *,
    method: AlignmentMethod,
    starting_checkpoint: dict[str, Any],
    teacher: dict[str, Any] | None,
    reference: dict[str, Any] | None,
    policy: dict[str, Any],
    metrics: AlignmentMetrics,
    cost: dict[str, Any],
    teacher_query_ratio: float | None,
    artifact_hashes: dict[str, str],
    limitations: list[str],
) -> dict[str, Any]:
    """Write paired JSON/Markdown cards without embedding local paths."""
    target = Path(destination)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "method": method.value,
        "starting_sft_checkpoint": starting_checkpoint,
        "teacher": teacher,
        "reference": reference,
        "policy": policy,
        "metrics": metrics.model_dump(mode="json"),
        "cost": cost,
        "teacher_query_ratio": teacher_query_ratio,
        "artifact_hashes": dict(sorted(artifact_hashes.items())),
        "limitations": limitations,
    }
    payload["card_sha256"] = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    json_target = target.with_suffix(".json")
    write_json_atomic(json_target, payload)
    metrics_json = json.dumps(payload["metrics"], indent=2, sort_keys=True)
    hashes = "\n".join(
        f"- `{name}`: `{digest}`" for name, digest in payload["artifact_hashes"].items()
    )
    caveats = "\n".join(f"- {item}" for item in limitations) or "- None recorded."
    markdown = f"""# Alignment Card

Method: `{method.value}`

Starting SFT checkpoint: `{starting_checkpoint.get("id", "unreported")}`

Teacher: `{(teacher or {}).get("id", "not applicable")}`

Reference: `{(reference or {}).get("id", "not applicable")}`

Policy dataset: `{policy.get("id", "unreported")}@{policy.get("revision", "unreported")}`

teacher-query ratio: `{teacher_query_ratio if teacher_query_ratio is not None else "not measured"}`

## Alignment, over-refusal and retained utility

```json
{metrics_json}
```

## Cost

```json
{json.dumps(cost, indent=2, sort_keys=True)}
```

## Artifact hashes

{hashes}

## Limitations

{caveats}

Card content digest: `{payload["card_sha256"]}`
"""
    write_text(target, markdown)
    return payload
