"""Lossless, validated Parquet exchange for the pinned verl prompt schema."""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any, Literal

from miniverl.errors import ConfigError, MissingDependencyError
from miniverl.utils.runs import write_json_atomic

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


def _report_path(parquet: Path) -> Path:
    return parquet.with_suffix(parquet.suffix + ".report.json")


def convert_dataset(
    source: str | Path,
    *,
    out: str | Path,
    direction: Direction,
    max_prompt_characters: int | None = None,
) -> dict[str, Any]:
    """Convert a Parquet dataset without truncation or semantic relabeling."""
    if direction not in {"from-verl-parquet", "to-verl-parquet"}:
        raise ConfigError(f"unknown dataset conversion direction {direction!r}")
    if max_prompt_characters is not None and max_prompt_characters < 1:
        raise ConfigError("max_prompt_characters must be positive")
    pa, pq = _pyarrow()
    source_path = Path(source)
    destination = Path(out)
    if not source_path.is_file():
        raise ConfigError(f"Parquet dataset not found: {source_path}")
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

        extension = row.pop("miniverl_extensions", None)
        if extension is None:
            extension = input_sidecar.get("rows", {}).get(str(source_index))
        if direction == "to-verl-parquet" and extension is not None:
            extra = dict(row.get("extra_info") or {})
            extra["miniverl"] = extension
            row["extra_info"] = extra
        elif direction == "from-verl-parquet":
            extra = dict(row.get("extra_info") or {})
            nested = extra.pop("miniverl", None)
            if nested is not None and extension is None:
                extension = nested
            # Parquet cannot encode an empty struct. ``None`` is the exact
            # canonical intermediate when all extension fields moved to the
            # checksummed sidecar; the reverse conversion restores the object.
            row["extra_info"] = extra or None
        if extension is not None:
            extensions[str(len(accepted))] = extension
        accepted.append(row)

    if not accepted:
        raise ConfigError("dataset conversion accepted zero rows", hint="inspect the rejected rows")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        pq.write_table(pa.Table.from_pylist(accepted), temporary)
        temporary.replace(destination)
    except Exception as exc:
        with suppress(OSError):
            temporary.unlink(missing_ok=True)
        raise ConfigError(f"cannot write Parquet dataset {destination}: {exc}") from exc

    extension_path = _sidecar_path(destination)
    if extensions and direction == "from-verl-parquet":
        write_json_atomic(
            extension_path,
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
        extension_path.unlink(missing_ok=True)

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
    report: dict[str, Any] = {
        "schema_version": 1,
        "direction": direction,
        "accepted_rows": len(accepted),
        "rejected_rows": len(rejected),
        "rejections": rejected,
        "truncation_risk": truncation,
        "source_sha256": _sha256(source_path),
        "output_sha256": _sha256(destination),
        "extension_namespace": "extra_info.miniverl",
        "extension_sidecar": str(extension_path) if extension_path.is_file() else None,
        "teacher_target_semantics": "distillation targets, not PPO reference log-probabilities",
    }
    write_json_atomic(_report_path(destination), report)
    return report
