"""Structural and provenance checks for exported verl bridge bundles."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import re
import struct
import sys
from pathlib import Path
from typing import Any

import yaml

from miniverl.bridge.contract import (
    BRIDGE_PROFILE,
    VERL_COMMIT,
    VERL_REPOSITORY,
    VERL_TAG,
)

__all__ = ["inspect_bridge_bundle"]

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


def _check_safetensors(path: Path) -> tuple[bool, str]:
    try:
        with path.open("rb") as handle:
            raw_length = handle.read(8)
            if len(raw_length) != 8:
                return False, "missing safetensors header length"
            header_length = struct.unpack("<Q", raw_length)[0]
            if header_length < 2 or header_length > path.stat().st_size - 8:
                return False, "invalid safetensors header length"
            header = json.loads(handle.read(header_length).decode("utf-8"))
        tensors = [key for key in header if key != "__metadata__"]
        if not tensors:
            return False, "safetensors contains no tensors"
        return True, f"{len(tensors)} tensor header(s)"
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, struct.error) as exc:
        return False, str(exc)


def _check_model(root: Path) -> dict[str, Any]:
    model = root / "model"
    config = model / "adapter_config.json"
    weights = model / "adapter_model.safetensors"
    try:
        payload = json.loads(config.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        model_check = {"status": "fail", "detail": str(exc)}
    else:
        weights_ok, detail = _check_safetensors(weights)
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
        model_check = {
            "status": "ok" if weights_ok and peft_ok else "fail",
            "peft_type": payload.get("peft_type"),
            "base_model": payload.get("base_model_name_or_path"),
            "detail": detail,
            "peft_config_load": peft_load,
            "load_scope": "PEFT config plus safetensors structure; base weights not loaded",
        }
    return model_check


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
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return {"status": "fail", "detail": "pyarrow is not installed"}
    required = {"data_source", "prompt", "ability", "reward_model", "extra_info"}
    schemas: dict[str, list[str]] = {}
    for split in ("train", "val"):
        path = root / "data" / f"{split}.parquet"
        try:
            table = pq.read_table(path)
        except Exception as exc:
            return {"status": "fail", "detail": f"{split}: {exc}"}
        names = set(table.schema.names)
        if not required.issubset(names) or table.num_rows < 1:
            return {
                "status": "fail",
                "detail": f"{split}: required={sorted(required)}, actual={sorted(names)}",
            }
        schemas[split] = table.schema.names
    return {"status": "ok", "schemas": schemas}


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


def _check_reward(root: Path) -> dict[str, Any]:
    path = root / "reward" / "reward_or_verifier_scaffold.py"
    try:
        spec = importlib.util.spec_from_file_location("_miniverl_exported_reward", path)
        if spec is None or spec.loader is None:
            raise ImportError("could not create an import specification")
        module = importlib.util.module_from_spec(spec)
        previous = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            spec.loader.exec_module(module)
        finally:
            sys.dont_write_bytecode = previous
        if not callable(getattr(module, "compute_score", None)):
            raise ImportError("compute_score is not callable")
    except Exception as exc:
        return {"status": "fail", "detail": str(exc)}
    source = path.read_text(encoding="utf-8")
    implementation_complete = "complete and test reward_or_verifier_scaffold" not in source
    return {
        "status": "ok",
        "detail": "side-effect-free import; scaffold intentionally not executed",
        "implementation_complete": implementation_complete,
    }


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


def _scan_dataset_text(
    root: Path,
    *,
    sentinels: tuple[str, ...],
    max_rows: int,
    max_bytes: int,
) -> dict[str, Any]:
    """Bounded, heuristic scan of string-like Parquet fields.

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
    truncated = False
    active = [(name, pattern) for name, pattern in _DATASET_DETECTORS]
    for path in files:
        split = path.stem
        try:
            table = pq.read_table(path)
        except Exception as exc:
            return {
                "status": "failed",
                "reason": f"cannot read {path.name}: {exc}",
                "findings": [],
            }
        rows_total += table.num_rows
        if truncated:
            continue
        for index, row in enumerate(table.to_pylist()):
            if rows_scanned >= max_rows or bytes_scanned >= max_bytes:
                truncated = True
                break
            rows_scanned += 1
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
        "rows_scanned": rows_scanned,
        "rows_total": rows_total,
        "bytes_scanned": bytes_scanned,
        "max_rows": max_rows,
        "max_bytes": max_bytes,
        "detectors": [name for name, _ in active] + (["user_sentinel"] if sentinels else []),
        "disclosure": "detector category, column and row index only; matched text is never reported",
        "findings": unique,
    }


def _check_privacy(
    root: Path,
    *,
    scan_dataset_text: bool = False,
    sentinels: tuple[str, ...] = (),
    max_rows: int = DATASET_SCAN_MAX_ROWS,
    max_bytes: int = DATASET_SCAN_MAX_BYTES,
) -> dict[str, Any]:
    """Report privacy per inspection scope; never widen one scope into another."""
    problems = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix in {".parquet", ".safetensors"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if _ABSOLUTE_PATH.search(text):
            problems.append(path.relative_to(root).as_posix())
    metadata_passed = not problems

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
        "portable_metadata_privacy": "passed" if metadata_passed else "failed",
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


def inspect_bridge_bundle(
    root: str | Path,
    *,
    require_verl: bool = False,
    require_tokenizer_load: bool = False,
    scan_dataset_text: bool = False,
    sentinels: tuple[str, ...] = (),
    dataset_scan_max_rows: int = DATASET_SCAN_MAX_ROWS,
    dataset_scan_max_bytes: int = DATASET_SCAN_MAX_BYTES,
) -> dict[str, Any]:
    """Return the complete bridge diagnosis; do not execute distributed code."""
    bundle = Path(root)
    target = _check_requirements(bundle)
    model = _check_model(bundle)
    tokenizer = _check_tokenizer(bundle, require_load=require_tokenizer_load)
    parquet = _check_parquet(bundle)
    config = _check_config(bundle)
    reward = _check_reward(bundle)
    hashes = _check_hashes(bundle)
    privacy = _check_privacy(
        bundle,
        scan_dataset_text=scan_dataset_text,
        sentinels=sentinels,
        max_rows=dataset_scan_max_rows,
        max_bytes=dataset_scan_max_bytes,
    )
    installed = _installed_verl()
    try:
        compatibility = json.loads(
            (bundle / "provenance" / "compatibility-report.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        compatibility = {}
    checks = (target, model, tokenizer, parquet, config, reward, hashes, privacy)
    artifact_failed = any(check.get("status") != "ok" for check in checks)
    pinned_verl_failed = require_verl and installed.get("status") != "ok"
    failed = artifact_failed or pinned_verl_failed
    local_smoke = "failed" if artifact_failed else "passed"
    return {
        "target_verl": target,
        "installed_verl": installed,
        "model_adapter_loadability": model,
        "tokenizer_identity": tokenizer,
        "tokenizer_verification_level": tokenizer["verification_level"],
        "portable_metadata_privacy": privacy["portable_metadata_privacy"],
        "dataset_content_privacy": privacy["dataset_content_privacy"],
        "model_weight_privacy": privacy["model_weight_privacy"],
        "parquet_schema": parquet,
        "config_profile": config,
        "reward_scaffold_importability": reward,
        "unsupported_semantics": compatibility.get("unsupported_semantics", []),
        "artifact_hashes": hashes,
        "privacy": privacy,
        "local_smoke_status": local_smoke,
        "artifact_bundle_complete": not artifact_failed,
        "upstream_config_parse_passed": bool(
            compatibility.get("upstream_config_parse_passed", False)
        ),
        "model_data_load_smoke_passed": bool(
            compatibility.get("model_data_load_smoke_passed", False)
        ),
        "reward_implementation_complete": bool(reward.get("implementation_complete", False)),
        "launchable": False,
        "distributed_execution_tested": bool(
            compatibility.get("distributed_execution_tested", False)
        ),
        "algorithm_semantic_parity": bool(compatibility.get("algorithm_semantic_parity", False)),
        "distributed_execution_status": compatibility.get(
            "distributed_execution_status", "not tested"
        ),
        "verdict": "fail" if failed else "ok",
    }
