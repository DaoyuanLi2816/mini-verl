"""Validated Parquet exchange for the pinned verl prompt schema.

Conversion is lossless for the rows it accepts and complete-or-nothing by
default: one invalid row fails the whole run rather than quietly publishing the
rest. ``allow_rejected_rows`` opts into a partial dataset, which the report
labels as incomplete instead of lossless.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from miniverl.bridge.publish import (
    DEFAULT_LOCK_TIMEOUT,
    OutputTransaction,
    dataset_output_targets,
    reject_source_output_alias,
)
from miniverl.errors import ConfigError, MissingDependencyError
from miniverl.utils.runs import canonical_json

__all__ = ["convert_dataset"]

Direction = Literal["from-verl-parquet", "to-verl-parquet"]


def _pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - dependency boundary
        raise MissingDependencyError("pyarrow", "bridge", "Parquet dataset conversion") from exc
    return pa, pq


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prompt_characters(prompt: list[Any]) -> int:
    return sum(
        len(str(message.get("content", ""))) for message in prompt if isinstance(message, Mapping)
    )


def _validate_row(row: Mapping[str, Any]) -> str | None:
    data_source = row.get("data_source")
    if not isinstance(data_source, str) or not data_source:
        return "data_source must be a non-empty string"
    prompt = row.get("prompt")
    if not isinstance(prompt, list) or not prompt:
        return "prompt must be a non-empty list of chat messages"
    for index, message in enumerate(prompt):
        if not isinstance(message, Mapping):
            return f"prompt[{index}] must be an object"
        if not isinstance(message.get("role"), str) or not message.get("role"):
            return f"prompt[{index}].role must be a non-empty string"
        if not isinstance(message.get("content"), str):
            return f"prompt[{index}].content must be a string"
    ability = row.get("ability")
    if ability is not None and not isinstance(ability, str):
        return "ability must be a string or null"
    reward_model = row.get("reward_model")
    if not isinstance(reward_model, Mapping) or reward_model.get("ground_truth") is None:
        return "reward_model.ground_truth is required"
    extra_info = row.get("extra_info")
    if extra_info is not None and not isinstance(extra_info, Mapping):
        return "extra_info must be an object or null"
    return None


def _sidecar_path(parquet: Path) -> Path:
    return parquet.with_suffix(parquet.suffix + ".miniverl.json")


def _write_parquet(pa: Any, pq: Any, rows: list[dict[str, Any]], path: Path) -> None:
    """Materialize the accepted rows. Seam kept module-level for fault injection."""
    pq.write_table(pa.Table.from_pylist(rows), path)


def _collect_extension_sources(
    row: dict[str, Any], *, source_index: int, input_sidecar: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Find every place this row's miniVERL extension can live.

    A row can carry extension data in three independent locations. Silently
    preferring one of them loses the others, so all of them are collected and
    reconciled by the caller.
    """
    found: dict[str, Any] = {}
    top_level = row.pop("miniverl_extensions", None)
    if top_level is not None:
        found["miniverl_extensions"] = top_level
    sidecar_rows = input_sidecar.get("rows")
    sidecar_value = (
        sidecar_rows.get(str(source_index)) if isinstance(sidecar_rows, Mapping) else None
    )
    if sidecar_value is not None:
        found["sidecar"] = sidecar_value
    extra = dict(row.get("extra_info") or {})
    nested = extra.pop("miniverl", None)
    if nested is not None:
        found["extra_info.miniverl"] = nested
    return found, extra


def _resolve_extension(found: Mapping[str, Any], *, source_index: int) -> tuple[Any, list[str]]:
    """Reconcile duplicate extension sources, failing closed on disagreement.

    Equal content from several locations is a deduplication, not a conflict.
    Different content cannot be resolved without guessing, so the conversion
    stops. Only the row index and the source *names* are reported: extension
    payloads can carry teacher targets and are never printed.
    """
    if not found:
        return None, []
    names = sorted(found)
    if len(names) == 1:
        return found[names[0]], []
    canonical = {name: canonical_json(value) for name, value in found.items()}
    if len(set(canonical.values())) > 1:
        raise ConfigError(
            f"row {source_index} carries conflicting miniVERL extension data in {', '.join(names)}",
            hint=(
                "these locations must agree or only one may be present; miniVERL will "
                "not guess which one is authoritative. Extension values are not shown "
                "here because they may contain teacher targets."
            ),
        )
    return found[names[0]], names


def convert_dataset(
    source: str | Path,
    *,
    out: str | Path,
    direction: Direction,
    max_prompt_characters: int | None = None,
    allow_rejected_rows: bool = False,
    overwrite: bool = False,
    lock_timeout: float = DEFAULT_LOCK_TIMEOUT,
) -> dict[str, Any]:
    """Convert a Parquet dataset without truncation or semantic relabeling.

    Conversion is complete-or-nothing: a row that fails validation fails the
    whole run unless ``allow_rejected_rows`` explicitly authorizes a partial
    dataset, which the report then labels as incomplete.
    """
    if direction not in {"from-verl-parquet", "to-verl-parquet"}:
        raise ConfigError(f"unknown dataset conversion direction {direction!r}")
    if max_prompt_characters is not None and max_prompt_characters < 1:
        raise ConfigError("max_prompt_characters must be positive")
    pa, pq = _pyarrow()
    source_path = Path(source)
    destination = Path(out)
    if not source_path.is_file():
        raise ConfigError(f"Parquet dataset not found: {source_path}")

    targets = dataset_output_targets(destination)
    # The source Parquet and its own sidecar are inputs; neither may be an
    # output of this conversion. Checked before the transaction is created.
    reject_source_output_alias(
        {
            "source Parquet": source_path,
            "source sidecar": _sidecar_path(source_path),
        },
        targets,
    )
    transaction = OutputTransaction(
        targets=targets,
        stem=destination.name,
        lock_root=destination.parent,
        overwrite=overwrite,
        lock_timeout=lock_timeout,
    )
    transaction.begin()
    try:
        return _convert_locked(
            transaction,
            pa,
            pq,
            source_path=source_path,
            targets=targets,
            direction=direction,
            max_prompt_characters=max_prompt_characters,
            allow_rejected_rows=allow_rejected_rows,
        )
    finally:
        transaction.close()


def _convert_locked(
    transaction: OutputTransaction,
    pa: Any,
    pq: Any,
    *,
    source_path: Path,
    targets: Mapping[str, Path],
    direction: Direction,
    max_prompt_characters: int | None,
    allow_rejected_rows: bool = False,
) -> dict[str, Any]:
    """Build and publish one coherent Parquet/sidecar/report family."""
    try:
        rows = pq.read_table(source_path).to_pylist()
    except Exception as exc:
        raise ConfigError(f"cannot read Parquet dataset {source_path}: {exc}") from exc

    input_sidecar: dict[str, Any] = {}
    candidate_sidecar = _sidecar_path(source_path)
    if candidate_sidecar.is_file():
        import json

        try:
            loaded = json.loads(candidate_sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigError(f"cannot read extension sidecar {candidate_sidecar}: {exc}") from exc
        if isinstance(loaded, dict):
            input_sidecar = loaded

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    extensions: dict[str, Any] = {}
    source_row_indices: dict[str, int] = {}
    deduplicated: list[dict[str, Any]] = []
    over_bound = 0
    for source_index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            rejected.append({"row": source_index, "reason": "row must be an object"})
            continue
        row = dict(raw)
        reason = _validate_row(row)
        if reason:
            rejected.append({"row": source_index, "reason": reason})
            continue
        prompt = row["prompt"]
        assert isinstance(prompt, list)
        if max_prompt_characters is not None and _prompt_characters(prompt) > max_prompt_characters:
            over_bound += 1

        found, extra = _collect_extension_sources(
            row, source_index=source_index, input_sidecar=input_sidecar
        )
        extension, merged_sources = _resolve_extension(found, source_index=source_index)
        if merged_sources:
            deduplicated.append({"row": source_index, "sources": merged_sources})
        if direction == "to-verl-parquet":
            if extension is not None:
                extra["miniverl"] = extension
            row["extra_info"] = extra or None
        else:
            # Parquet cannot encode an empty struct. ``None`` is the exact
            # canonical intermediate when all extension fields moved to the
            # checksummed sidecar; the reverse conversion restores the object.
            row["extra_info"] = extra or None
        if extension is not None:
            # Keyed on the output row so the sidecar stays consistent with the
            # Parquet file it accompanies; the source index is kept separately.
            extensions[str(len(accepted))] = extension
        source_row_indices[str(len(accepted))] = source_index
        accepted.append(row)

    if rejected and not allow_rejected_rows:
        first = rejected[0]
        raise ConfigError(
            f"{len(rejected)} of {len(rows)} source row(s) failed validation; "
            f"first failure is row {first['row']}: {first['reason']}",
            hint=(
                "conversion is complete-or-nothing by default. Fix the source rows, or "
                "pass --allow-rejected-rows to publish an explicitly partial dataset"
            ),
        )
    if not accepted:
        raise ConfigError("dataset conversion accepted zero rows", hint="inspect the rejected rows")

    staged_parquet = transaction.path("parquet")
    try:
        _write_parquet(pa, pq, accepted, staged_parquet)
    except Exception as exc:
        raise ConfigError(f"cannot write Parquet dataset {targets['parquet']}: {exc}") from exc
    transaction.claim("parquet")

    emit_sidecar = bool(extensions) and direction == "from-verl-parquet"
    if emit_sidecar:
        transaction.write_json(
            "sidecar",
            {
                "schema_version": 1,
                "namespace": "extra_info.miniverl",
                "semantics": (
                    "miniVERL token provenance and teacher targets; never PPO reference log-probabilities"
                ),
                "rows": extensions,
            },
        )
    else:
        # A previous conversion's sidecar must not outlive the run that replaced it.
        transaction.discard("sidecar")

    truncation = (
        {"status": "not_evaluated_no_tokenizer"}
        if max_prompt_characters is None
        else {
            "status": "character_bound_only",
            "max_prompt_characters": max_prompt_characters,
            "rows_over_bound": over_bound,
            "rows_truncated": 0,
        }
    )
    partial = bool(rejected)
    report: dict[str, Any] = {
        "schema_version": 3,
        "direction": direction,
        "source_rows": len(rows),
        "accepted_rows": len(accepted),
        "rejected_rows": len(rejected),
        "rejections": rejected,
        # A conversion that dropped rows is never described as lossless overall.
        "complete_dataset_conversion": not partial,
        "lossless_for_accepted_rows": True,
        "partial_conversion": partial,
        "partial_conversion_authorized": bool(allow_rejected_rows) if partial else None,
        # Output row -> original source row, so a partial file keeps provenance.
        "source_row_indices": source_row_indices if partial else None,
        "extension_deduplication": deduplicated,
        "truncation_risk": truncation,
        "source_sha256": _sha256(source_path),
        # Hashed from the staged bytes that this same transaction publishes.
        "output_sha256": _sha256(staged_parquet),
        "extension_namespace": "extra_info.miniverl",
        "extension_sidecar": str(targets["sidecar"]) if emit_sidecar else None,
        "report_path": targets["report"].name,
        "teacher_target_semantics": "distillation targets, not PPO reference log-probabilities",
    }
    transaction.write_json("report", report)
    transaction.commit()
    return report
