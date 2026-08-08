"""Prepare, validate and report an external alignment evaluation suite.

``prepare`` is the step that freezes the study. It resolves each endpoint at
its pinned revision, selects the task subset deterministically from benchmark
metadata alone, and writes a manifest whose digest every later step is checked
against. Selection never looks at a model outcome, so a suite cannot be tuned
to flatter a method.

``validate`` re-checks a finished result set against the manifest it claims to
come from. ``report`` aggregates task rows into the published metrics.

Nothing here downloads during ``evaluate``: the suite is prepared once, with
network access, and evaluated offline.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from miniverl.alignment_external.records import validate_rows
from miniverl.alignment_external.registry import load_registry

__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "prepare_suite",
    "report_suite",
    "select_task_ids",
    "validate_results",
]

MANIFEST_SCHEMA_VERSION = 1

#: Upper bound from the compute contract: no model generates more than this.
MAX_TASKS_PER_MODEL = 512


def _digest(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def select_task_ids(
    task_ids: Sequence[str],
    strata: Sequence[str] | None,
    *,
    limit: int,
    seed: int,
) -> list[str]:
    """Deterministically choose ``limit`` task ids, preserving category coverage.

    Selection uses benchmark metadata only -- ids and stratum labels. It never
    sees a model output, so it cannot be steered toward a favourable subset.
    Sampling is round-robin across strata so a small subset keeps every
    category rather than over-representing the largest one.
    """
    if limit < 1:
        raise ValueError("limit must be positive")
    ordered = sorted(task_ids)
    if len(ordered) <= limit:
        return ordered
    if strata is None:
        rng = random.Random(seed)
        return sorted(rng.sample(ordered, limit))

    if len(strata) != len(task_ids):
        raise ValueError("strata must be parallel to task_ids")
    buckets: dict[str, list[str]] = {}
    for task_id, stratum in zip(task_ids, strata, strict=True):
        buckets.setdefault(str(stratum), []).append(str(task_id))

    chosen: list[str] = []
    for name in sorted(buckets):
        rng = random.Random(f"{seed}:{name}")
        rng.shuffle(buckets[name])
    # Round robin across strata until the budget is spent.
    index = 0
    while len(chosen) < limit:
        progressed = False
        for name in sorted(buckets):
            bucket = buckets[name]
            if index < len(bucket):
                chosen.append(bucket[index])
                progressed = True
                if len(chosen) == limit:
                    break
        if not progressed:
            break
        index += 1
    return sorted(chosen)


def prepare_suite(
    *,
    profile: Mapping[str, Any],
    out: str | Path,
    registry_path: str | Path | None = None,
    resolver: Any = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Freeze one evaluation suite and write its manifest.

    ``resolver(endpoint) -> (task_ids, strata)`` supplies the upstream task ids;
    it is injected so the offline fixtures can exercise this without a network
    call.
    """
    registry = load_registry(registry_path)
    by_id = {entry["id"]: entry for entry in registry["endpoints"]}
    requested = profile.get("endpoints") or []
    if not requested:
        raise ValueError("the profile selects no endpoints")

    endpoints: list[dict[str, Any]] = []
    total_generation_tasks = 0
    for item in requested:
        endpoint_id = str(item["id"])
        # `external: false` marks a miniVERL-internal measurement -- the
        # retained tool-utility side of every Pareto comparison. It has no
        # upstream dataset revision or licence, so it is not in the endpoint
        # registry, but its task ids are frozen in the same manifest so the
        # utility and alignment sides of a comparison use one fixed suite.
        is_external = bool(item.get("external", True))
        entry = by_id.get(endpoint_id)
        if is_external and entry is None:
            raise ValueError(
                f"profile names endpoint {endpoint_id!r}, which is not in the registry"
            )
        if entry is None:
            entry = {
                "id": endpoint_id,
                "category": str(item.get("category", "retained_utility")),
                "dataset": None,
                "revision": None,
                "config": None,
                "split": None,
                "evaluator": {
                    "kind": "deterministic_rule",
                    "implementation": "miniverl internal evaluation",
                    "model": None,
                },
                "strata_field": None,
            }
        limit = int(item["tasks"])
        if limit < 1:
            raise ValueError(f"endpoint {endpoint_id}: tasks must be positive")

        task_ids, strata = resolver(entry)
        selected = select_task_ids(
            task_ids, strata, limit=limit, seed=int(profile["selection_seed"])
        )
        if len(selected) < min(limit, len(task_ids)):
            raise ValueError(f"endpoint {endpoint_id}: selection returned too few tasks")
        if item.get("counts_toward_generation", True):
            total_generation_tasks += len(selected)

        endpoints.append(
            {
                "id": endpoint_id,
                "category": entry["category"],
                "dataset": entry["dataset"],
                "revision": entry["revision"],
                "config": entry.get("config"),
                "split": entry["split"],
                "evaluator": entry["evaluator"],
                "external": is_external,
                "requested_tasks": limit,
                "selected_tasks": len(selected),
                "task_ids": selected,
                "task_ids_digest": _digest(selected),
                "strata_field": entry.get("strata_field"),
            }
        )

    if total_generation_tasks > MAX_TASKS_PER_MODEL:
        raise ValueError(
            f"the profile generates {total_generation_tasks} tasks per model, above the "
            f"{MAX_TASKS_PER_MODEL} compute-contract ceiling"
        )

    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "profile_id": str(profile["id"]),
        "selection_seed": int(profile["selection_seed"]),
        "selection_rule": (
            "round robin across strata after a per-stratum seeded shuffle; "
            "benchmark metadata only, never a model outcome"
        ),
        "generation_tasks_per_model": total_generation_tasks,
        "max_tasks_per_model": MAX_TASKS_PER_MODEL,
        "endpoints": endpoints,
    }
    manifest["manifest_digest"] = _digest(
        {key: value for key, value in manifest.items() if key != "manifest_digest"}
    )

    if not dry_run:
        destination = Path(out)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "suite-manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return manifest


def validate_results(manifest: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Every reason a result set does not match the suite it claims."""
    problems = validate_rows([dict(row) for row in rows])

    expected = {entry["id"]: entry for entry in manifest["endpoints"]}
    seen_by_endpoint: dict[str, set[str]] = {}
    for row in rows:
        endpoint_id = str(row.get("endpoint_id"))
        entry = expected.get(endpoint_id)
        if entry is None:
            problems.append(f"result names endpoint {endpoint_id!r}, absent from the manifest")
            continue
        if str(row.get("dataset_revision")) != entry["revision"]:
            problems.append(
                f"{endpoint_id}/{row.get('task_id')}: dataset revision "
                f"{row.get('dataset_revision')!r} is not the pinned {entry['revision']!r}"
            )
        seen_by_endpoint.setdefault(endpoint_id, set()).add(str(row.get("task_id")))

    for endpoint_id, entry in expected.items():
        seen = seen_by_endpoint.get(endpoint_id, set())
        planned = set(entry["task_ids"])
        missing = planned - seen
        extra = seen - planned
        if missing:
            problems.append(f"{endpoint_id}: {len(missing)} planned task(s) have no result row")
        if extra:
            problems.append(f"{endpoint_id}: {len(extra)} result row(s) are not in the suite")
    return problems


def report_suite(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate task rows per endpoint without inventing a combined score."""
    by_endpoint: dict[str, dict[str, Any]] = {}
    for row in rows:
        endpoint_id = str(row["endpoint_id"])
        bucket = by_endpoint.setdefault(
            endpoint_id,
            {
                "category": row.get("category"),
                "evaluated": 0,
                "not_applicable": 0,
                "failed": 0,
                "score_sum": 0.0,
                "not_applicable_reasons": {},
            },
        )
        status = str(row.get("status"))
        if status == "evaluated":
            bucket["evaluated"] += 1
            bucket["score_sum"] += float(row["score"])
        elif status == "not_applicable":
            bucket["not_applicable"] += 1
            reason = str(row.get("not_applicable_reason", "unstated"))
            bucket["not_applicable_reasons"][reason] = (
                bucket["not_applicable_reasons"].get(reason, 0) + 1
            )
        else:
            bucket["failed"] += 1

    endpoints: dict[str, Any] = {}
    for endpoint_id, bucket in sorted(by_endpoint.items()):
        evaluated = bucket["evaluated"]
        endpoints[endpoint_id] = {
            "category": bucket["category"],
            "tasks_evaluated": evaluated,
            "tasks_not_applicable": bucket["not_applicable"],
            "tasks_failed": bucket["failed"],
            # None, never 0.0, when nothing was measured.
            "mean_score": (bucket["score_sum"] / evaluated) if evaluated else None,
            "not_applicable_reasons": dict(sorted(bucket["not_applicable_reasons"].items())),
        }
    return {
        "endpoints": endpoints,
        "note": (
            "per-endpoint means only. There is no combined alignment score: the "
            "endpoints measure different things and move independently"
        ),
    }
