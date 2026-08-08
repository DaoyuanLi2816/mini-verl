"""Structural and provenance checks for exported verl bridge bundles."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from miniverl.bridge.contract import (
    BRIDGE_PROFILE,
    VERL_COMMIT,
    VERL_REPOSITORY,
    VERL_TAG,
)
from miniverl.bridge.preflight import preflight_bundle_tree
from miniverl.bridge.reward_static import REWARD_LEVELS, inspect_reward_scaffold
from miniverl.bridge.safetensors_check import SAFETENSORS_LEVELS, inspect_safetensors

__all__ = [
    "REWARD_LEVELS",
    "SAFETENSORS_LEVELS",
    "TOKENIZER_LEVELS",
    "inspect_bridge_bundle",
]

_ABSOLUTE_PATH = re.compile(
    r"(?i)(?:(?<![A-Za-z0-9])[A-Z]:[\\/]|/home/|/users/|\\\\[^\\]+\\[^\\]+)"
)

# ------------------------------------------------------------------ tokenizer

#: Ordered weakest to strongest. Presence of files is *not* a load.
TOKENIZER_LEVELS = (
    "not_present",
    "metadata_only",
    "loadable_local_snapshot",
    "structural_identity_verified",
)

#: Any one of these makes a local snapshot loadable; ``tokenizer_config.json``
#: alone is metadata and proves nothing about the vocabulary.
_TOKENIZER_VOCABULARY = (
    ("tokenizer.json",),
    ("vocab.json", "merges.txt"),
    ("tokenizer.model",),
    ("vocab.txt",),
)

#: Tokenizer files whose name does not contain "tokenizer".
_TOKENIZER_SIDE_FILES = frozenset(
    {"vocab.json", "vocab.txt", "merges.txt", "special_tokens_map.json", "added_tokens.json"}
)

# ---------------------------------------------------------------- privacy

#: Heuristic detectors. Categories are reported; matched text never is.
_DATASET_DETECTORS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("url_userinfo", re.compile(r"[a-zA-Z][a-zA-Z0-9+.\-]*://[^/\s:@]+:[^/\s@]+@")),
    ("private_key_block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("aws_access_key_id", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("bearer_token", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{20,}")),
    (
        "credential_assignment",
        re.compile(
            r"(?i)\b(?:api[_-]?key|secret[_-]?key|secret|password|passwd|"
            r"access[_-]?token|auth[_-]?token|token)\b\s*[:=]\s*\S"
        ),
    ),
    ("absolute_local_path", _ABSOLUTE_PATH),
)

DATASET_SCAN_MAX_ROWS = 1000
DATASET_SCAN_MAX_BYTES = 8 * 1024 * 1024
_DATASET_SCAN_MAX_DEPTH = 6
#: Rows decoded at once. Small enough that a single huge row group cannot be
#: materialized just to honour a much smaller row bound.
_DATASET_SCAN_BATCH_ROWS = 256


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_requirements(root: Path) -> dict[str, Any]:
    path = root / "recipe" / "REQUIRED_VERL.txt"
    expected = {
        "VERL_REPOSITORY": VERL_REPOSITORY,
        "VERL_TAG": VERL_TAG,
        "VERL_COMMIT": VERL_COMMIT,
        "PROFILE": BRIDGE_PROFILE,
    }
    try:
        values = dict(line.split("=", 1) for line in path.read_text(encoding="utf-8").splitlines())
    except (OSError, ValueError) as exc:
        return {"status": "fail", "detail": str(exc)}
    mismatches = {
        key: values.get(key) for key, value in expected.items() if values.get(key) != value
    }
    return {
        "status": "ok" if not mismatches else "fail",
        "tag": values.get("VERL_TAG"),
        "commit": values.get("VERL_COMMIT"),
        "profile": values.get("PROFILE"),
        "mismatches": mismatches,
    }


def _check_model(root: Path, *, require_payload: bool = False) -> dict[str, Any]:
    model = root / "model"
    config = model / "adapter_config.json"
    weights = model / "adapter_model.safetensors"
    try:
        payload = json.loads(config.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "fail", "detail": str(exc)}

    tensors = inspect_safetensors(weights, require_payload=require_payload)
    weights_ok = tensors["status"] == "ok"
    peft_ok = str(payload.get("peft_type", "")).upper() == "LORA"
    peft_load = "not installed; structural validation only"
    if weights_ok and peft_ok:
        try:
            from peft import PeftConfig

            loaded = PeftConfig.from_pretrained(str(model))
            peft_load = f"loaded {type(loaded).__name__}"
        except ImportError:
            pass
        except Exception as exc:
            weights_ok = False
            peft_load = f"PEFT rejected adapter_config.json: {exc}"
    return {
        "status": "ok" if weights_ok and peft_ok else "fail",
        "peft_type": payload.get("peft_type"),
        "base_model": payload.get("base_model_name_or_path"),
        "detail": tensors["detail"],
        "safetensors": tensors,
        "safetensors_verification_level": tensors["verification_level"],
        "peft_config_load": peft_load,
        "load_scope": (
            "PEFT config plus safetensors payload structure; base model weights are "
            "never loaded and tensor values are never interpreted"
        ),
    }


def _reference_tokenizer_identity(root: Path) -> dict[str, Any]:
    """Tokenizer identity recorded by the source run, if the bundle carries one."""
    try:
        manifest = json.loads(
            (root / "provenance" / "miniverl-manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return {}
    identity = manifest.get("tokenizer_identity")
    return identity if isinstance(identity, dict) else {}


def _tokenizer_components(model: Path) -> tuple[list[Path], list[str], bool]:
    """Return tokenizer files, missing component names and vocabulary presence."""
    files = (
        sorted(
            path
            for path in model.iterdir()
            if path.is_file() and ("tokenizer" in path.name or path.name in _TOKENIZER_SIDE_FILES)
        )
        if model.is_dir()
        else []
    )
    present = {path.name for path in files}
    has_vocabulary = any(set(group).issubset(present) for group in _TOKENIZER_VOCABULARY)
    missing: list[str] = []
    if not has_vocabulary:
        missing.append(
            "vocabulary (tokenizer.json, vocab.json+merges.txt, tokenizer.model or vocab.txt)"
        )
    if "tokenizer_config.json" not in present:
        missing.append("tokenizer_config.json")
    if "special_tokens_map.json" not in present:
        missing.append("special_tokens_map.json")
    return files, missing, has_vocabulary


def _check_tokenizer(root: Path, *, require_load: bool) -> dict[str, Any]:
    """Distinguish tokenizer metadata presence from a verified local load.

    Filename and digest presence is never reported as tokenizer compatibility.
    Loading is strictly local: ``local_files_only=True`` and
    ``trust_remote_code=False``, so no network call and no remote code path
    exists here.
    """
    model = root / "model"
    files, missing, has_vocabulary = _tokenizer_components(model)

    file_digest = hashlib.sha256()
    for path in files:
        file_digest.update(path.name.encode("utf-8"))
        file_digest.update(_sha256(path).encode("ascii"))

    check: dict[str, Any] = {
        "status": "ok" if files else "fail",
        "verification_level": "not_present" if not files else "metadata_only",
        "files": [path.name for path in files],
        "missing_components": missing,
        "file_digest": file_digest.hexdigest() if files else None,
        "structural_identity": None,
        "reference_identity_source": None,
        "load_attempt": "not_attempted",
        "network_access": "never; local_files_only=True and trust_remote_code=False",
        "scope": (
            "file presence and content digest only; this is not proof that the "
            "tokenizer loads or matches the source run"
        ),
        "mismatches": [],
    }
    if not files:
        check["load_attempt"] = "not_attempted: no tokenizer file is present"
        check["detail"] = "the bundle carries no tokenizer metadata"
        return _finalize_tokenizer(check, require_load=require_load)
    if not has_vocabulary:
        check["load_attempt"] = "not_attempted: tokenizer vocabulary is absent"
        return _finalize_tokenizer(check, require_load=require_load)
    try:
        from transformers import AutoTokenizer
    except ImportError:
        check["load_attempt"] = "not_attempted: transformers is not installed"
        return _finalize_tokenizer(check, require_load=require_load)

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(model), local_files_only=True, trust_remote_code=False
        )
    except Exception as exc:
        check["status"] = "fail"
        check["load_attempt"] = f"failed: {exc}"
        return _finalize_tokenizer(check, require_load=require_load)

    from miniverl.models.tokenizers import tokenizer_structural_digest

    check["load_attempt"] = "passed"
    check["verification_level"] = "loadable_local_snapshot"
    check["scope"] = "loaded from local files only; identity compared where a reference exists"
    structural = tokenizer_structural_digest(tokenizer)
    special = {
        str(name): str(value)
        for name, value in sorted(getattr(tokenizer, "special_tokens_map", {}).items())
    }
    check["structural_identity"] = {
        "structural_digest_v2": structural,
        "vocab_size": len(tokenizer),
        "special_tokens_map": special,
        "tokenizer_class": type(tokenizer).__name__,
    }

    reference = _reference_tokenizer_identity(root)
    if not reference:
        check["reference_identity_source"] = "none recorded in the bundle manifest"
        return _finalize_tokenizer(check, require_load=require_load)

    check["reference_identity_source"] = "provenance/miniverl-manifest.json"
    mismatches: list[str] = []
    expected_digest = reference.get("structural_digest_v2")
    if expected_digest and expected_digest != structural:
        mismatches.append("structural_digest_v2")
    expected_vocab = reference.get("vocab_size")
    if isinstance(expected_vocab, int) and expected_vocab != len(tokenizer):
        mismatches.append("vocab_size")
    expected_special = reference.get("special_tokens_map")
    if isinstance(expected_special, dict):
        normalized = {str(name): str(value) for name, value in sorted(expected_special.items())}
        if normalized != special:
            mismatches.append("special_tokens_map")
    check["mismatches"] = mismatches
    if mismatches:
        check["status"] = "fail"
    elif expected_digest:
        check["verification_level"] = "structural_identity_verified"
        check["scope"] = "loaded locally and structurally identical to the recorded source run"
    return _finalize_tokenizer(check, require_load=require_load)


def _finalize_tokenizer(check: dict[str, Any], *, require_load: bool) -> dict[str, Any]:
    level = check["verification_level"]
    check["strict_load_required"] = require_load
    check["strict_load_satisfied"] = level in {
        "loadable_local_snapshot",
        "structural_identity_verified",
    }
    if require_load and not check["strict_load_satisfied"]:
        check["status"] = "fail"
    return check


def _check_parquet(root: Path) -> dict[str, Any]:
    """Validate the exchange schema from Parquet metadata alone.

    Column names, row counts and row-group counts all live in the footer, so a
    verl-scale dataset must never be materialized to answer them.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return {"status": "fail", "detail": "pyarrow is not installed"}
    required = {"data_source", "prompt", "ability", "reward_model", "extra_info"}
    schemas: dict[str, list[str]] = {}
    rows: dict[str, int] = {}
    for split in ("train", "val"):
        path = root / "data" / f"{split}.parquet"
        try:
            handle = pq.ParquetFile(path)
        except Exception as exc:
            return {"status": "fail", "detail": f"{split}: {exc}"}
        try:
            names = list(handle.schema_arrow.names)
            num_rows = handle.metadata.num_rows
        finally:
            handle.close()
        if not required.issubset(set(names)) or num_rows < 1:
            return {
                "status": "fail",
                "detail": f"{split}: required={sorted(required)}, actual={sorted(names)}",
            }
        schemas[split] = names
        rows[split] = num_rows
    return {
        "status": "ok",
        "schemas": schemas,
        "rows": rows,
        "read_scope": "Parquet footer metadata only; no row group was decoded",
    }


def _check_config(root: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(
            (root / "recipe" / "verl-overrides.yaml").read_text(encoding="utf-8")
        )
        adapter = json.loads((root / "model" / "adapter_config.json").read_text(encoding="utf-8"))
        base = json.loads((root / "model" / "base-model.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
        return {"status": "fail", "detail": str(exc)}
    required_roots = {"data", "actor_rollout_ref", "trainer", "custom_reward_function"}
    actual = set(payload) if isinstance(payload, dict) else set()
    problems: list[str] = []
    if not required_roots.issubset(actual):
        problems.append("missing required root")
    try:
        model = payload["actor_rollout_ref"]["model"]
        expected = {
            "path": "model/base",
            "lora_adapter_path": "model",
            "lora_rank": adapter["r"],
            "lora_alpha": adapter["lora_alpha"],
            "target_modules": adapter["target_modules"],
        }
        for field, value in expected.items():
            if model.get(field) != value:
                problems.append(f"actor_rollout_ref.model.{field}")
        if base.get("model_id") != adapter["base_model_name_or_path"]:
            problems.append("base-model.json model_id")
        if base.get("revision") != adapter["revision"]:
            problems.append("base-model.json revision")
        if base.get("materialized_path") != "model/base":
            problems.append("base-model.json materialized_path")
    except (KeyError, TypeError):
        problems.append("invalid model handoff structure")
    return {
        "status": "ok" if not problems else "fail",
        "profile": BRIDGE_PROFILE,
        "roots": sorted(actual),
        "model_handoff_problems": problems,
    }


def _check_reward(root: Path, *, trust_and_import: bool = False) -> dict[str, Any]:
    """Statically verify the reward interface.

    The bundle is untrusted input. Inspection parses the scaffold and never
    imports it, so a bundle cannot act merely by being diagnosed.
    """
    path = root / "reward" / "reward_or_verifier_scaffold.py"
    return inspect_reward_scaffold(path, trust_and_import=trust_and_import)


def _check_hashes(root: Path) -> dict[str, Any]:
    checksum = root / "provenance" / "SHA256SUMS"
    try:
        declared = {}
        for line in checksum.read_text(encoding="utf-8").splitlines():
            digest, relative = line.split("  ", 1)
            declared[relative] = digest
    except (OSError, ValueError) as exc:
        return {"status": "fail", "detail": str(exc)}
    actual_files = {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if path.is_file() and path != checksum
    }
    problems = []
    for relative, path in actual_files.items():
        if declared.get(relative) != _sha256(path):
            problems.append(relative)
    problems.extend(sorted(set(declared) - set(actual_files)))
    return {
        "status": "ok" if not problems and set(declared) == set(actual_files) else "fail",
        "files": len(actual_files),
        "problems": sorted(set(problems)),
    }


def _iter_strings(value: Any, depth: int = 0) -> Any:
    """Yield string leaves of a decoded Parquet cell, bounded by depth."""
    if depth > _DATASET_SCAN_MAX_DEPTH:
        return
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_strings(item, depth + 1)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_strings(item, depth + 1)


def _has_string_leaf(data_type: Any) -> bool:
    """Whether an Arrow type can contain a string anywhere inside it."""
    import pyarrow as pa

    if pa.types.is_string(data_type) or pa.types.is_large_string(data_type):
        return True
    if pa.types.is_struct(data_type):
        return any(_has_string_leaf(field.type) for field in data_type)
    if pa.types.is_map(data_type):
        return _has_string_leaf(data_type.key_type) or _has_string_leaf(data_type.item_type)
    if hasattr(data_type, "value_type") and data_type.num_fields:
        return _has_string_leaf(data_type.value_type)
    return False


def _string_like_columns(schema: Any) -> list[str]:
    """Columns worth decoding; everything else cannot hold a secret as text."""
    return [field.name for field in schema if _has_string_leaf(field.type)]


def _scan_dataset_text(
    root: Path,
    *,
    sentinels: tuple[str, ...],
    max_rows: int,
    max_bytes: int,
) -> dict[str, Any]:
    """Bounded, streaming, heuristic scan of string-like Parquet fields.

    The bounds are enforced *while reading*: row groups are pulled one at a
    time and decoding stops the moment ``max_rows`` or ``max_bytes`` is
    reached, so a verl-scale dataset is never materialized. Row counts for
    unread files still come from the footer, which costs nothing.

    This is a detector, not de-identification proof. Matched text is never
    returned or logged: only the detector category, the column and the row
    index leave this function. ``.safetensors`` is never read as text.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return {
            "status": "not_inspected",
            "reason": "pyarrow is not installed",
            "findings": [],
        }
    files = sorted(path for path in (root / "data").glob("*.parquet") if path.is_file())
    if not files:
        return {"status": "not_inspected", "reason": "no Parquet file found", "findings": []}

    findings: list[dict[str, Any]] = []
    rows_scanned = 0
    rows_total = 0
    bytes_scanned = 0
    row_groups_read = 0
    files_inspected = 0
    truncated = False
    active = list(_DATASET_DETECTORS)

    def _bounds_reached() -> bool:
        return rows_scanned >= max_rows or bytes_scanned >= max_bytes

    for path in files:
        split = path.stem
        try:
            handle = pq.ParquetFile(path)
        except Exception as exc:
            return {
                "status": "failed",
                "reason": f"cannot read {path.name}: {exc}",
                "findings": [],
            }
        try:
            rows_total += handle.metadata.num_rows
            if truncated:
                # Totals still come from the footer; no row group is decoded.
                continue
            columns = _string_like_columns(handle.schema_arrow)
            if not columns:
                continue
            files_inspected += 1
            file_row_index = 0
            for group_index in range(handle.num_row_groups):
                if _bounds_reached():
                    truncated = True
                    break
                row_groups_read += 1
                try:
                    batches = handle.iter_batches(
                        batch_size=_DATASET_SCAN_BATCH_ROWS,
                        row_groups=[group_index],
                        columns=columns,
                    )
                    for batch in batches:
                        if _bounds_reached():
                            truncated = True
                            break
                        for offset, row in enumerate(batch.to_pylist()):
                            if _bounds_reached():
                                truncated = True
                                break
                            rows_scanned += 1
                            index = file_row_index + offset
                            if not isinstance(row, dict):
                                continue
                            for column, cell in row.items():
                                for text in _iter_strings(cell):
                                    bytes_scanned += len(text.encode("utf-8", errors="ignore"))
                                    for category, pattern in active:
                                        if pattern.search(text):
                                            findings.append(
                                                {
                                                    "category": category,
                                                    "split": split,
                                                    "column": str(column),
                                                    "row": index,
                                                }
                                            )
                                    for sentinel in sentinels:
                                        if sentinel and sentinel in text:
                                            findings.append(
                                                {
                                                    "category": "user_sentinel",
                                                    "split": split,
                                                    "column": str(column),
                                                    "row": index,
                                                }
                                            )
                        file_row_index += batch.num_rows
                except Exception as exc:
                    return {
                        "status": "failed",
                        "reason": f"cannot read row group {group_index} of {path.name}: {exc}",
                        "findings": [],
                    }
        finally:
            handle.close()
    # Deduplicate so one noisy column cannot flood the report.
    unique: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for item in findings:
        key = (item["category"], item["split"], item["column"], item["row"])
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return {
        "status": "failed" if unique else "passed",
        "method": "heuristic regular-expression detectors; not de-identification proof",
        "scan_scope": "sampled" if truncated else "full",
        "files_total": len(files),
        "files_inspected": files_inspected,
        "row_groups_read": row_groups_read,
        "rows_scanned": rows_scanned,
        "rows_total": rows_total,
        "bytes_scanned": bytes_scanned,
        "max_rows": max_rows,
        "max_bytes": max_bytes,
        "batch_rows": _DATASET_SCAN_BATCH_ROWS,
        "detectors": [name for name, _ in active] + (["user_sentinel"] if sentinels else []),
        "disclosure": "detector category, column and row index only; matched text is never reported",
        "findings": unique,
    }


#: Bounds for the portable-metadata scan. A bundle must not be able to stall a
#: diagnosis by shipping an enormous "metadata" file.
METADATA_MAX_FILE_BYTES = 1_000_000
METADATA_MAX_TOTAL_BYTES = 32 * 1024 * 1024
METADATA_MAX_FINDINGS = 200

#: Structured keys whose *name* means the value is a credential, whatever it
#: looks like. Matched against the final path component only.
_SECRET_KEY_NAME = re.compile(
    r"(?i)^(api[_-]?key|secret([_-]?key)?|password|passwd|access[_-]?token|"
    r"auth([_-]?token)?|token|authorization|credentials?|private[_-]?key|"
    r"client[_-]?secret|session[_-]?key)$"
)

#: Value-shaped detectors. These run against strings wherever they are found.
_VALUE_DETECTORS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("url_userinfo", re.compile(r"[a-zA-Z][a-zA-Z0-9+.\-]*://[^/\s:@]+:[^/\s@]+@")),
    ("private_key_block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("aws_access_key_id", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("bearer_token", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{20,}")),
    ("absolute_local_path", _ABSOLUTE_PATH),
)

#: Text-shaped detector. Only meaningful in unstructured prose/scripts, where
#: there is no key/value structure to inspect.
_TEXT_CREDENTIAL = re.compile(
    r"(?i)\b(?:api[_-]?key|secret[_-]?key|secret|password|passwd|"
    r"access[_-]?token|auth[_-]?token|api[_-]?token|token)\b\s*[:=]\s*\S"
)

_STRUCTURED_SUFFIXES = {".json", ".yaml", ".yml"}
_NEVER_TEXT_SUFFIXES = {".parquet", ".safetensors", ".bin", ".pt", ".onnx", ".npz"}


def _scan_string(value: str, *, sentinels: tuple[str, ...]) -> list[str]:
    """Detector categories matched by one string. Never returns the text."""
    categories = [name for name, pattern in _VALUE_DETECTORS if pattern.search(value)]
    if any(sentinel and sentinel in value for sentinel in sentinels):
        categories.append("user_sentinel")
    return categories


def _walk_structured(
    value: Any,
    *,
    path: str,
    relative: str,
    sentinels: tuple[str, ...],
    findings: list[dict[str, Any]],
    depth: int = 0,
) -> None:
    """Report credential-shaped keys and values with their JSON path only."""
    if depth > 12 or len(findings) > METADATA_MAX_FINDINGS:
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{path}.{key}"
            if _SECRET_KEY_NAME.match(str(key)) and isinstance(item, str) and item.strip():
                findings.append(
                    {
                        "category": "semantic_secret_key",
                        "file": relative,
                        "path": child,
                        "detail": "a key whose name denotes a credential holds a non-empty value",
                    }
                )
            _walk_structured(
                item,
                path=child,
                relative=relative,
                sentinels=sentinels,
                findings=findings,
                depth=depth + 1,
            )
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _walk_structured(
                item,
                path=f"{path}[{index}]",
                relative=relative,
                sentinels=sentinels,
                findings=findings,
                depth=depth + 1,
            )
    elif isinstance(value, str):
        for category in _scan_string(value, sentinels=sentinels):
            findings.append(
                {
                    "category": category,
                    "file": relative,
                    "path": path,
                    "detail": "a detector matched this value",
                }
            )


def _scan_portable_metadata(root: Path, *, sentinels: tuple[str, ...]) -> dict[str, Any]:
    """Bounded heuristic scan of the bundle's portable text metadata.

    Structured files are parsed so a finding can name a JSON path; everything
    else is scanned line by line. Matched text never leaves this function --
    only the file, location and detector category do.
    """
    findings: list[dict[str, Any]] = []
    files_inspected = 0
    files_skipped_too_large = 0
    bytes_scanned = 0
    truncated = False
    # Anything the scan could not look at. "No finding" over an incomplete
    # inspection is not the same statement as "no finding".
    gaps: list[dict[str, str]] = []

    def _gap(relative: str, reason: str) -> None:
        nonlocal truncated
        truncated = True
        if len(gaps) < METADATA_MAX_FINDINGS:
            gaps.append({"file": relative, "reason": reason})

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() in _NEVER_TEXT_SUFFIXES:
            continue
        relative = path.relative_to(root).as_posix()
        if bytes_scanned >= METADATA_MAX_TOTAL_BYTES:
            _gap(relative, "total_byte_limit_reached")
            break
        if len(findings) > METADATA_MAX_FINDINGS:
            _gap(relative, "finding_limit_reached")
            break
        try:
            size = path.stat().st_size
        except OSError as exc:
            _gap(relative, f"stat_failed: {type(exc).__name__}")
            continue
        if size > METADATA_MAX_FILE_BYTES:
            files_skipped_too_large += 1
            _gap(relative, "file_larger_than_scan_limit")
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            _gap(relative, "not_decodable_as_utf8")
            continue
        except OSError as exc:
            _gap(relative, f"read_failed: {type(exc).__name__}")
            continue
        files_inspected += 1
        bytes_scanned += len(text.encode("utf-8", errors="ignore"))

        parsed: Any = None
        if path.suffix.lower() in _STRUCTURED_SUFFIXES:
            try:
                parsed = (
                    json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
                )
            except (json.JSONDecodeError, yaml.YAMLError):
                parsed = None
        if parsed is not None:
            _walk_structured(
                parsed, path="$", relative=relative, sentinels=sentinels, findings=findings
            )
            continue

        for number, line in enumerate(text.splitlines(), start=1):
            categories = _scan_string(line, sentinels=sentinels)
            if _TEXT_CREDENTIAL.search(line):
                categories.append("credential_assignment")
            for category in categories:
                findings.append(
                    {
                        "category": category,
                        "file": relative,
                        "line": number,
                        "detail": "a detector matched this line",
                    }
                )

    unique: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for item in findings:
        key = (item["category"], item["file"], item.get("path"), item.get("line"))
        if key not in seen:
            seen.add(key)
            unique.append(item)
    if unique:
        status = "heuristic_failed"
    elif truncated:
        # Nothing was found, but not everything was looked at. Reporting this
        # as "passed" would turn an incomplete inspection into a clean bill.
        status = "heuristic_incomplete"
    else:
        status = "heuristic_passed_full"
    return {
        "status": status,
        "method": (
            "heuristic key-name and regular-expression detectors; not de-identification proof"
        ),
        "scan_scope": "sampled" if truncated else "full",
        "complete": not truncated,
        "incomplete_reasons": gaps,
        "files_inspected": files_inspected,
        "files_skipped_too_large": files_skipped_too_large,
        "bytes_scanned": bytes_scanned,
        "max_file_bytes": METADATA_MAX_FILE_BYTES,
        "max_total_bytes": METADATA_MAX_TOTAL_BYTES,
        "max_findings": METADATA_MAX_FINDINGS,
        "findings_truncated": len(unique) > METADATA_MAX_FINDINGS,
        "disclosure": (
            "file, JSON path or line number and detector category only; "
            "matched text is never reported"
        ),
        "findings": unique[:METADATA_MAX_FINDINGS],
    }


def _check_privacy(
    root: Path,
    *,
    scan_dataset_text: bool = False,
    sentinels: tuple[str, ...] = (),
    max_rows: int = DATASET_SCAN_MAX_ROWS,
    max_bytes: int = DATASET_SCAN_MAX_BYTES,
    require_complete_metadata_scan: bool = False,
) -> dict[str, Any]:
    """Report privacy per inspection scope; never widen one scope into another."""
    metadata = _scan_portable_metadata(root, sentinels=sentinels)
    problems = sorted({finding["file"] for finding in metadata["findings"]})
    # An incomplete scan found nothing, which is weaker than finding nothing.
    # By default that is reported but not failed; strict mode refuses it.
    metadata_complete = metadata["status"] == "heuristic_passed_full"
    metadata_passed = metadata_complete or (
        metadata["status"] == "heuristic_incomplete" and not require_complete_metadata_scan
    )

    if scan_dataset_text:
        dataset = _scan_dataset_text(
            root, sentinels=sentinels, max_rows=max_rows, max_bytes=max_bytes
        )
    else:
        dataset = {
            "status": "not_inspected",
            "reason": "pass --scan-dataset-text to run the bounded heuristic scan",
            "findings": [],
        }
    dataset_status = dataset["status"]
    return {
        # ``ok``/``fail`` drives the overall verdict and never reflects a scope
        # that was not inspected.
        "status": "ok" if metadata_passed and dataset_status != "failed" else "fail",
        "portable_metadata_privacy": metadata["status"],
        "portable_metadata_scan_complete": metadata_complete,
        "strict_metadata_scan_required": require_complete_metadata_scan,
        "metadata_scan": metadata,
        "dataset_content_privacy": dataset_status,
        "model_weight_privacy": "not_inspected",
        "model_weight_reason": (
            "safetensors payloads are never read as text and no meaningful weight "
            "privacy check exists; this scope is not evaluated"
        ),
        "scope_note": (
            "only portable metadata files are inspected by default; "
            "not_inspected never means passed"
        ),
        "problems": problems,
        "dataset_scan": dataset,
    }


def _installed_verl() -> dict[str, Any]:
    try:
        version = importlib.metadata.version("verl")
        distribution = importlib.metadata.distribution("verl")
    except importlib.metadata.PackageNotFoundError:
        return {"status": "not installed", "version": None, "direct_url": None}
    direct_url = None
    direct_url_text = distribution.read_text("direct_url.json")
    if direct_url_text:
        try:
            direct_url = json.loads(direct_url_text)
        except json.JSONDecodeError:
            direct_url = {"invalid": True}
    commit = ((direct_url or {}).get("vcs_info") or {}).get("commit_id")
    return {
        "status": "ok" if commit == VERL_COMMIT else "unverified",
        "version": version,
        "direct_url": direct_url,
        "expected_commit": VERL_COMMIT,
    }


#: Facts a bundle can only *assert*. Nothing in a doctor run recomputes them:
#: they describe events that happened elsewhere, earlier, on other hardware.
_DECLARED_ONLY_CLAIMS = (
    "upstream_config_parse_passed",
    "model_data_load_smoke_passed",
    "distributed_execution_tested",
    "algorithm_semantic_parity",
    "launchable",
    "distributed_execution_status",
)


def _bundle_declared_claims(compatibility: dict[str, Any], *, present: bool) -> dict[str, Any]:
    """Everything the bundle says about itself, labelled as its own testimony."""
    declared: dict[str, Any] = {
        "source": "provenance/compatibility-report.json" if present else "absent",
        "trust": "unsigned_self_consistent" if present else "not_verified",
        "note": (
            "these values are copied from the bundle and describe events this doctor run "
            "did not observe; they are claims, not results"
        ),
    }
    for name in _DECLARED_ONLY_CLAIMS:
        if name in compatibility:
            declared[name] = compatibility[name]
    declared["unsupported_semantics"] = compatibility.get("unsupported_semantics", [])
    return declared


def _locally_recomputed_checks(
    *,
    target: dict[str, Any],
    model: dict[str, Any],
    tokenizer: dict[str, Any],
    parquet: dict[str, Any],
    config: dict[str, Any],
    reward: dict[str, Any],
    hashes: dict[str, Any],
    privacy: dict[str, Any],
    installed: dict[str, Any],
    upstream: dict[str, Any],
) -> dict[str, Any]:
    """Only checks this process performed against the bytes on disk."""

    def _verdict(check: dict[str, Any]) -> str:
        return "passed" if check.get("status") == "ok" else "failed"

    return {
        "checksum_consistency": _verdict(hashes),
        "pinned_requirement_file": _verdict(target),
        "config_structure": _verdict(config),
        "adapter_safetensors_structure": _verdict(model),
        "tokenizer_identity": _verdict(tokenizer),
        "parquet_schema": _verdict(parquet),
        "reward_interface_static": _verdict(reward),
        "portable_metadata_privacy": (
            "passed"
            if privacy["portable_metadata_privacy"] == "heuristic_passed_full"
            else "failed"
        ),
        "installed_verl_identity": installed.get("status", "not installed"),
        "upstream_config_parse": upstream["status"],
        # A doctor run never launches a job and never compares algorithms.
        "distributed_execution": "not_run",
        "algorithm_semantic_parity": "not_run",
    }


def _provenance_trust(hashes: dict[str, Any]) -> dict[str, Any]:
    """State exactly what the bundle's own checksum file can and cannot prove."""
    consistent = hashes.get("status") == "ok"
    return {
        "level": "unsigned_self_consistent" if consistent else "not_verified",
        "signature_verification": "not_available",
        "note": (
            "SHA256SUMS ships inside the bundle it describes, so agreement proves "
            "internal consistency only; anyone who edits a file can regenerate it. "
            "miniVERL implements no signature or transparency-log verification."
        ),
    }


def _recompute_upstream_smoke(
    root: Path, installed: dict[str, Any], *, enabled: bool
) -> dict[str, Any]:
    """Re-run the documented upstream parse/merge locally, in this process.

    ``--require-verl`` used to compare only an installed commit id, which said
    nothing about whether this bundle's config still merges into the pinned
    upstream schema. This performs the merge now rather than trusting the
    bundle's record of a merge someone else performed.
    """
    if not enabled:
        return {"status": "not_run", "reason": "pass --require-verl to recompute the local smoke"}
    if installed.get("status") == "not installed":
        return {"status": "failed", "reason": "the pinned verl distribution is not installed"}
    try:
        from omegaconf import OmegaConf
    except ImportError:
        return {"status": "failed", "reason": "omegaconf is not installed"}
    try:
        import importlib.metadata

        distribution = importlib.metadata.distribution("verl")
        generated = Path(
            str(distribution.locate_file("verl/trainer/config/_generated_ppo_trainer.yaml"))
        )
        if not generated.is_file():
            return {"status": "failed", "reason": "installed verl omits the generated PPO config"}
        official = OmegaConf.load(generated)
        exported = OmegaConf.load(root / "recipe" / "verl-overrides.yaml")
        OmegaConf.set_struct(official, True)
        OmegaConf.merge(official, exported)
    except Exception as exc:
        return {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
    return {
        "status": "passed",
        "scope": (
            "the bundle's verl-overrides.yaml parsed and merged into the installed pinned "
            "upstream schema in this process; no distributed job was launched"
        ),
    }


def _preflight_refusal(tree: dict[str, Any]) -> dict[str, Any]:
    """Refuse the bundle without having read any of its content.

    Every per-check field is reported as ``not_inspected`` rather than
    ``failed``: the checks did not run and did not find anything wrong. The one
    established fact is that the tree is not a plain directory of regular files.
    """
    reasons = sorted({item["reason"] for item in tree["rejections"]})
    return {
        "bundle_tree_preflight": tree,
        "target_verl": {"status": "not_inspected"},
        "installed_verl": _installed_verl(),
        "model_adapter_loadability": {"status": "not_inspected"},
        "safetensors_verification_level": "not_inspected",
        "tokenizer_identity": {"status": "not_inspected"},
        "tokenizer_verification_level": "not_inspected",
        "portable_metadata_privacy": "not_inspected",
        "dataset_content_privacy": "not_inspected",
        "model_weight_privacy": "not_inspected",
        "parquet_schema": {"status": "not_inspected"},
        "config_profile": {"status": "not_inspected"},
        "reward_scaffold_importability": {"status": "not_inspected"},
        "reward_verification_level": "not_inspected",
        "unsupported_semantics": [],
        "artifact_hashes": {"status": "not_inspected"},
        "privacy": {"status": "not_inspected"},
        "bundle_declared_claims": {
            "status": "not_inspected",
            "note": "the bundle tree was refused before any declaration was read",
        },
        "locally_recomputed_checks": {},
        "provenance_trust": {"level": "not_verified", "reason": "bundle tree refused"},
        "local_smoke_status": "not_run",
        "artifact_bundle_complete": False,
        "upstream_config_parse_passed": False,
        "model_data_load_smoke_passed": False,
        "reward_implementation_complete": False,
        "launchable": False,
        "distributed_execution_tested": False,
        "algorithm_semantic_parity": False,
        "distributed_execution_status": "not tested",
        "detail": ("refused before reading any bundle content: " + ", ".join(reasons)),
        "verdict": "fail",
    }


def inspect_bridge_bundle(
    root: str | Path,
    *,
    require_verl: bool = False,
    require_tokenizer_load: bool = False,
    require_adapter_payload: bool = False,
    trust_and_import_reward_code: bool = False,
    scan_dataset_text: bool = False,
    require_complete_metadata_scan: bool = False,
    sentinels: tuple[str, ...] = (),
    dataset_scan_max_rows: int = DATASET_SCAN_MAX_ROWS,
    dataset_scan_max_bytes: int = DATASET_SCAN_MAX_BYTES,
) -> dict[str, Any]:
    """Return the complete bridge diagnosis.

    No code from the inspected bundle runs unless ``trust_and_import_reward_code``
    is explicitly set, and no distributed job is ever launched.
    """
    bundle = Path(root)
    # Every check below opens a path inside the bundle, and an open follows
    # whatever that path resolves to. Validate the tree shape first, so a
    # hostile bundle cannot point an entry at a file outside itself and have it
    # hashed or searched for credentials.
    tree = preflight_bundle_tree(bundle)
    if tree["status"] != "ok":
        return _preflight_refusal(tree)
    target = _check_requirements(bundle)
    model = _check_model(bundle, require_payload=require_adapter_payload)
    tokenizer = _check_tokenizer(bundle, require_load=require_tokenizer_load)
    parquet = _check_parquet(bundle)
    config = _check_config(bundle)
    reward = _check_reward(bundle, trust_and_import=trust_and_import_reward_code)
    hashes = _check_hashes(bundle)
    privacy = _check_privacy(
        bundle,
        scan_dataset_text=scan_dataset_text,
        sentinels=sentinels,
        max_rows=dataset_scan_max_rows,
        max_bytes=dataset_scan_max_bytes,
        require_complete_metadata_scan=require_complete_metadata_scan,
    )
    installed = _installed_verl()
    compatibility_path = bundle / "provenance" / "compatibility-report.json"
    try:
        compatibility = json.loads(compatibility_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        compatibility = {}
    checks = (target, model, tokenizer, parquet, config, reward, hashes, privacy)
    artifact_failed = any(check.get("status") != "ok" for check in checks)
    upstream = _recompute_upstream_smoke(bundle, installed, enabled=require_verl)
    pinned_verl_failed = require_verl and (
        installed.get("status") != "ok" or upstream["status"] != "passed"
    )
    failed = artifact_failed or pinned_verl_failed
    local_smoke = "failed" if artifact_failed else "passed"
    declared = _bundle_declared_claims(compatibility, present=compatibility_path.is_file())
    recomputed = _locally_recomputed_checks(
        target=target,
        model=model,
        tokenizer=tokenizer,
        parquet=parquet,
        config=config,
        reward=reward,
        hashes=hashes,
        privacy=privacy,
        installed=installed,
        upstream=upstream,
    )
    return {
        "target_verl": target,
        "installed_verl": installed,
        "model_adapter_loadability": model,
        "safetensors_verification_level": model.get(
            "safetensors_verification_level", "not_present"
        ),
        "tokenizer_identity": tokenizer,
        "tokenizer_verification_level": tokenizer["verification_level"],
        "portable_metadata_privacy": privacy["portable_metadata_privacy"],
        "dataset_content_privacy": privacy["dataset_content_privacy"],
        "model_weight_privacy": privacy["model_weight_privacy"],
        "parquet_schema": parquet,
        "config_profile": config,
        "reward_scaffold_interface": reward,
        "reward_verification_level": reward["verification_level"],
        "reward_code_executed": bool(reward.get("code_executed", False)),
        # Copied from the bundle for the reader's information; the doctor did not
        # verify that this list is complete.
        "unsupported_semantics": compatibility.get("unsupported_semantics", []),
        "artifact_hashes": hashes,
        "privacy": privacy,
        "local_smoke_status": local_smoke,
        "artifact_bundle_complete": not artifact_failed,
        "bundle_declared_claims": declared,
        "locally_recomputed_checks": recomputed,
        "provenance_trust": _provenance_trust(hashes),
        "upstream_config_parse_recheck": upstream,
        # Every flag below reflects what *this* process recomputed. A bundle
        # cannot raise any of them by describing itself favourably.
        "upstream_config_parse_passed": upstream["status"] == "passed",
        "model_data_load_smoke_passed": upstream["status"] == "passed"
        and not artifact_failed
        and require_verl,
        "reward_implementation_complete": bool(reward.get("implementation_complete", False)),
        "launchable": False,
        "distributed_execution_tested": False,
        "algorithm_semantic_parity": False,
        "distributed_execution_status": "not tested",
        "verdict": "fail" if failed else "ok",
    }
