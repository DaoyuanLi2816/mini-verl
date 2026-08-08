"""Validated Parquet exchange for the pinned verl prompt schema.

Conversion is lossless for the rows it accepts and complete-or-nothing by
default: one invalid row fails the whole run rather than quietly publishing the
rest. ``allow_rejected_rows`` opts into a partial dataset, which the report
labels as incomplete instead of lossless.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from miniverl import __version__
from miniverl.bridge.publish import (
    DEFAULT_LOCK_TIMEOUT,
    OutputTransaction,
    dataset_output_targets,
    reject_source_output_alias,
)
from miniverl.errors import ConfigError, MissingDependencyError
from miniverl.utils.runs import canonical_json

__all__ = ["convert_dataset", "resolve_source_row"]

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


def _write_parquet(pa: Any, writer: Any, rows: list[dict[str, Any]], schema: Any) -> None:
    """Write one buffered batch. Seam kept module-level for fault injection."""
    writer.write_table(pa.Table.from_pylist(rows, schema=schema))


#: Rows decoded from Parquet at a time, and the write buffer size. Small enough
#: that a huge dataset never becomes a single Python list.
CONVERSION_BATCH_ROWS = 512

#: A partial conversion can reject an unbounded number of rows. Counts stay
#: exact; the per-row detail array is sampled so the report cannot grow without
#: limit. Provenance for accepted rows is never sampled.
MAX_REPORTED_REJECTIONS = 100

#: v1 is what 0.6.0-0.6.3 published: it *may* carry a digest binding. v2 is
#: what this version writes, and the binding is mandatory, so a sidecar can no
#: longer be copied next to an unrelated Parquet file and silently believed.
#: Field names stay `source_sha256`/`source_rows` rather than gaining a second
#: `dataset_*` spelling for the same concept -- one validator, one vocabulary.
_SIDECAR_SCHEMA_VERSIONS = frozenset({1, 2})
_SIDECAR_WRITE_SCHEMA_VERSION = 2
_SIDECAR_BOUND_SCHEMA_VERSIONS = frozenset({2})
_SIDECAR_NAMESPACE = "extra_info.miniverl"

#: Any other top-level key is treated as an unknown critical field. A sidecar
#: this version does not fully understand must not be half-applied.
_SIDECAR_KNOWN_FIELDS = frozenset(
    {
        "schema_version",
        "namespace",
        "semantics",
        "rows",
        # Optional binding a sidecar may carry to name the dataset it belongs to.
        "source_sha256",
        "source_rows",
        "generator",
    }
)


def _sidecar_error(path: Path, detail: str) -> ConfigError:
    """Sidecar diagnostics name the location, never the extension payload."""
    return ConfigError(
        f"extension sidecar {path.name} is invalid: {detail}",
        hint=(
            "a sidecar that exists but cannot be validated is never treated as absent, "
            "because that would silently drop miniVERL extension provenance. Fix or "
            "remove the sidecar. Extension values are not shown here because they may "
            "contain teacher targets."
        ),
    )


def _load_input_sidecar(path: Path, *, source_rows: int) -> dict[str, Any]:
    """Validate an existing sidecar strictly, or fail closed.

    Up to the v0.6.3 release candidate any JSON object was accepted, so ``{}``
    and ``{"rows": []}`` both read as "no extensions" and quietly discarded the
    provenance the sidecar existed to carry.
    """
    if not path.is_file():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise _sidecar_error(path, f"it cannot be read as JSON ({type(exc).__name__})") from exc
    if not isinstance(loaded, Mapping):
        raise _sidecar_error(path, "the top level must be a JSON object")

    unknown = sorted(set(loaded) - _SIDECAR_KNOWN_FIELDS)
    if unknown:
        raise _sidecar_error(path, f"unknown field(s) {', '.join(unknown)}")
    version = loaded.get("schema_version")
    if version not in _SIDECAR_SCHEMA_VERSIONS:
        supported = ", ".join(str(item) for item in sorted(_SIDECAR_SCHEMA_VERSIONS))
        raise _sidecar_error(path, f"schema_version {version!r} is not one of {supported}")
    namespace = loaded.get("namespace")
    if namespace != _SIDECAR_NAMESPACE:
        raise _sidecar_error(path, f"namespace must be {_SIDECAR_NAMESPACE!r}, got {namespace!r}")
    rows = loaded.get("rows")
    if not isinstance(rows, Mapping):
        raise _sidecar_error(path, "rows must be a JSON object keyed by source row index")

    declared_rows = loaded.get("source_rows")
    if version in _SIDECAR_BOUND_SCHEMA_VERSIONS:
        # A v2 sidecar promises to name its dataset. Accepting one without the
        # binding would reintroduce exactly the ambiguity v2 exists to remove.
        missing = [field for field in ("source_sha256", "source_rows") if loaded.get(field) is None]
        if missing:
            raise _sidecar_error(
                path,
                f"schema_version {version} requires {', '.join(missing)} to bind it to one dataset",
            )
    if declared_rows is not None and declared_rows != source_rows:
        raise _sidecar_error(
            path, f"source_rows {declared_rows!r} does not match the dataset's {source_rows}"
        )

    for key in rows:
        text = str(key)
        if not text.isdigit() or str(int(text)) != text:
            raise _sidecar_error(
                path, f"row key {text!r} is not a canonical non-negative integer string"
            )
        if int(text) >= source_rows:
            raise _sidecar_error(
                path, f"row key {text!r} is outside the source dataset's {source_rows} row(s)"
            )
    try:
        canonical_json(dict(rows))
    except (TypeError, ValueError) as exc:
        raise _sidecar_error(path, "row values are not JSON-compatible") from exc
    return dict(loaded)


def resolve_source_row(runs: Sequence[Mapping[str, int]] | None, output_row: int) -> int | None:
    """Map an output row back to its source row through the run encoding.

    ``runs`` is a report's ``source_row_runs``. ``None`` -- a complete
    conversion -- means the mapping is the identity. Returns ``None`` when the
    output row is outside every run.
    """
    if runs is None:
        return output_row
    for run in runs:
        offset = output_row - run["output_start"]
        if 0 <= offset < run["length"]:
            return run["source_start"] + offset
    return None


def _source_identity(path: Path) -> dict[str, Any]:
    """Cheap identity of the source file: what it is and how big it was."""
    try:
        stat = path.stat()
    except OSError as exc:
        raise ConfigError(f"cannot stat source dataset {path}: {exc}") from exc
    return {
        # st_ino/st_dev are 0 on some Windows filesystems; size and mtime still
        # move when a file is replaced, and the digest below is authoritative.
        "inode": stat.st_ino,
        "device": stat.st_dev,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _assert_source_unchanged(path: Path, before: Mapping[str, Any], digest: str) -> None:
    """Refuse to publish a report describing bytes the conversion did not read."""
    after = _source_identity(path)
    changed = sorted(field for field, value in before.items() if after.get(field) != value)
    if changed:
        raise ConfigError(
            f"source dataset {path.name} changed during conversion ({', '.join(changed)})",
            hint=(
                "the conversion read one file and would have published a report "
                "describing another. Nothing was published. Re-run against a "
                "source that is not being written concurrently."
            ),
        )
    if _sha256(path) != digest:
        raise ConfigError(
            f"source dataset {path.name} changed during conversion (content digest)",
            hint=(
                "the file has the same size and timestamp but different bytes. "
                "Nothing was published."
            ),
        )


def _verify_sidecar_binding(sidecar: Mapping[str, Any], path: Path, *, source: Path) -> None:
    """Check the optional digest binding after the cheap structural checks."""
    declared = sidecar.get("source_sha256")
    if declared is not None and declared != _sha256(source):
        raise _sidecar_error(path, "source_sha256 does not match the dataset being converted")


def _extra_info_type(pa: Any, source_type: Any, *, direction: str, extension_type: Any) -> Any:
    """Output type for ``extra_info`` after the miniVERL field moves."""
    fields = []
    if source_type is not None and pa.types.is_struct(source_type):
        fields = [field for field in source_type if field.name != "miniverl"]
    if direction == "to-verl-parquet" and extension_type is not None:
        fields.append(pa.field("miniverl", extension_type))
    if not fields:
        # Parquet cannot encode an empty struct; an all-null column is the exact
        # canonical intermediate.
        return pa.null()
    return pa.struct(fields)


def _output_schema(pa: Any, source_schema: Any, *, direction: str, extension_type: Any) -> Any:
    """Derive one output schema up front so every batch writes the same shape.

    Inferring a schema per batch would let an optional nested field that only
    appears in a later row group produce an incompatible second schema.
    """
    fields = []
    has_extra_info = False
    for field in source_schema:
        if field.name == "miniverl_extensions":
            # Never an output column: it is reconciled into the sidecar or into
            # ``extra_info.miniverl``.
            continue
        if field.name == "extra_info":
            has_extra_info = True
            fields.append(
                pa.field(
                    "extra_info",
                    _extra_info_type(
                        pa, field.type, direction=direction, extension_type=extension_type
                    ),
                )
            )
        else:
            fields.append(field)
    if not has_extra_info and direction == "to-verl-parquet" and extension_type is not None:
        fields.append(pa.field("extra_info", pa.struct([pa.field("miniverl", extension_type)])))
    return pa.schema(fields)


def _extension_type(pa: Any, source_schema: Any, sidecar: Mapping[str, Any]) -> Any:
    """Arrow type for the miniVERL extension payload, or ``None`` if there is none."""
    for field in source_schema:
        if field.name == "miniverl_extensions":
            return field.type
    rows = sidecar.get("rows") or {}
    if not rows:
        return None
    try:
        return pa.array(list(rows.values())).type
    except (pa.ArrowInvalid, pa.ArrowTypeError, TypeError):  # pragma: no cover - exotic payloads
        return None


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
    """Build and publish one coherent Parquet/sidecar/report family.

    The source is streamed a record batch at a time and accepted rows are
    written straight into the staging file, so neither the whole table nor the
    whole converted dataset is ever held in memory. Publication still happens
    only in ``transaction.commit()``, which keeps strict conversion
    complete-or-nothing.
    """
    # Identity of the source as it was when conversion started. A conversion
    # streams the file over a long period and then publishes a report claiming
    # a `source_sha256`; if the file is replaced in between, that claim would
    # describe bytes the conversion never read.
    source_identity_before = _source_identity(source_path)

    try:
        parquet_file = pq.ParquetFile(source_path)
        source_schema = parquet_file.schema_arrow
        source_rows = parquet_file.metadata.num_rows
    except Exception as exc:
        raise ConfigError(f"cannot read Parquet dataset {source_path}: {exc}") from exc

    candidate_sidecar = _sidecar_path(source_path)
    input_sidecar = _load_input_sidecar(candidate_sidecar, source_rows=source_rows)
    if input_sidecar:
        _verify_sidecar_binding(input_sidecar, candidate_sidecar, source=source_path)

    extension_type = _extension_type(pa, source_schema, input_sidecar)
    schema = _output_schema(pa, source_schema, direction=direction, extension_type=extension_type)

    rejected: list[dict[str, Any]] = []
    rejected_total = 0
    extensions: dict[str, Any] = {}
    # Output row -> source row as contiguous runs rather than one entry per
    # row. A complete conversion is one run and the mapping is the identity;
    # each rejected row starts at most one new run, so this is O(rejections).
    remap: list[dict[str, int]] = []
    deduplicated: list[dict[str, Any]] = []
    deduplicated_total = 0
    over_bound = 0
    accepted_total = 0
    buffer: list[dict[str, Any]] = []

    staged_parquet = transaction.path("parquet")
    writer = pq.ParquetWriter(staged_parquet, schema)

    def _flush() -> None:
        if not buffer:
            return
        _write_parquet(pa, writer, buffer, schema)
        buffer.clear()

    try:
        source_index = -1
        for batch in parquet_file.iter_batches(batch_size=CONVERSION_BATCH_ROWS):
            for raw in batch.to_pylist():
                source_index += 1
                if not isinstance(raw, Mapping):
                    reason = "row must be an object"
                elif (validation := _validate_row(dict(raw))) is not None:
                    reason = validation
                else:
                    reason = None
                if reason is not None:
                    # Fail before reading any further row group unless a partial
                    # dataset was explicitly authorized.
                    if not allow_rejected_rows:
                        raise ConfigError(
                            f"source row {source_index} failed validation: {reason}",
                            hint=(
                                "conversion is complete-or-nothing by default. Fix the source "
                                "rows, or pass --allow-rejected-rows to publish an explicitly "
                                "partial dataset"
                            ),
                        )
                    rejected_total += 1
                    if len(rejected) < MAX_REPORTED_REJECTIONS:
                        rejected.append({"row": source_index, "reason": reason})
                    continue

                row = dict(raw)
                prompt = row["prompt"]
                assert isinstance(prompt, list)
                if (
                    max_prompt_characters is not None
                    and _prompt_characters(prompt) > max_prompt_characters
                ):
                    over_bound += 1

                found, extra = _collect_extension_sources(
                    row, source_index=source_index, input_sidecar=input_sidecar
                )
                extension, merged_sources = _resolve_extension(found, source_index=source_index)
                if merged_sources:
                    deduplicated_total += 1
                    if len(deduplicated) < MAX_REPORTED_REJECTIONS:
                        deduplicated.append({"row": source_index, "sources": merged_sources})
                if direction == "to-verl-parquet" and extension is not None:
                    extra["miniverl"] = extension
                # Parquet cannot encode an empty struct. ``None`` is the exact
                # canonical intermediate when all extension fields moved to the
                # checksummed sidecar; the reverse conversion restores the object.
                row["extra_info"] = extra or None
                if extension is not None:
                    # Keyed on the output row so the sidecar stays consistent with
                    # the Parquet file it accompanies.
                    extensions[str(accepted_total)] = extension
                if remap and remap[-1]["source_start"] + remap[-1]["length"] == source_index:
                    remap[-1]["length"] += 1
                else:
                    remap.append(
                        {
                            "output_start": accepted_total,
                            "source_start": source_index,
                            "length": 1,
                        }
                    )
                accepted_total += 1
                buffer.append(row)
                if len(buffer) >= CONVERSION_BATCH_ROWS:
                    _flush()
        _flush()
    except ConfigError:
        raise
    except Exception as exc:
        raise ConfigError(f"cannot write Parquet dataset {targets['parquet']}: {exc}") from exc
    finally:
        writer.close()

    if not accepted_total:
        raise ConfigError("dataset conversion accepted zero rows", hint="inspect the rejected rows")

    # Everything is staged and nothing is published yet, so this is the last
    # point at which the whole family can still be discarded.
    source_digest = _sha256(source_path)
    _assert_source_unchanged(source_path, source_identity_before, source_digest)

    transaction.claim("parquet")

    emit_sidecar = bool(extensions) and direction == "from-verl-parquet"
    if emit_sidecar:
        # Bind the sidecar to the exact Parquet it is published beside, hashed
        # from the staged bytes this transaction is about to commit. Without
        # this a sidecar could be copied next to an unrelated dataset and its
        # row keys silently reinterpreted against different rows.
        transaction.write_json(
            "sidecar",
            {
                "schema_version": _SIDECAR_WRITE_SCHEMA_VERSION,
                "namespace": _SIDECAR_NAMESPACE,
                "semantics": (
                    "miniVERL token provenance and teacher targets; never PPO reference log-probabilities"
                ),
                "source_sha256": _sha256(staged_parquet),
                "source_rows": accepted_total,
                "generator": {"name": "miniverl", "version": __version__},
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
    partial = bool(rejected_total)
    report: dict[str, Any] = {
        "schema_version": 3,
        "direction": direction,
        "source_rows": source_rows,
        "accepted_rows": accepted_total,
        "rejected_rows": rejected_total,
        # Counts stay exact; only the per-row detail is sampled so one very
        # broken dataset cannot produce an unbounded report.
        "rejections": rejected,
        "rejections_truncated": rejected_total > len(rejected),
        "max_reported_rejections": MAX_REPORTED_REJECTIONS,
        "conversion_batch_rows": CONVERSION_BATCH_ROWS,
        # A conversion that dropped rows is never described as lossless overall.
        "complete_dataset_conversion": not partial,
        "lossless_for_accepted_rows": True,
        "partial_conversion": partial,
        "partial_conversion_authorized": bool(allow_rejected_rows) if partial else None,
        # Output row -> original source row, so a partial file keeps full
        # provenance. Contiguous runs, not one entry per row: a complete
        # conversion is a single identity run and each rejection starts at
        # most one more.
        "source_row_runs": remap if partial else None,
        "source_row_mapping_encoding": (
            "contiguous runs; output row output_start+k maps to source row "
            "source_start+k for k in [0, length)"
        ),
        "extension_deduplication": deduplicated,
        "extension_deduplication_total": deduplicated_total,
        "extension_deduplication_truncated": deduplicated_total > len(deduplicated),
        "truncation_risk": truncation,
        "source_sha256": source_digest,
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
