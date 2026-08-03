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


def _check_model(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
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
    tokenizer_files = (
        sorted(path for path in model.iterdir() if path.is_file() and "tokenizer" in path.name)
        if model.is_dir()
        else []
    )
    tokenizer_digest = hashlib.sha256()
    for path in tokenizer_files:
        tokenizer_digest.update(path.name.encode("utf-8"))
        tokenizer_digest.update(_sha256(path).encode("ascii"))
    tokenizer_check = {
        "status": "ok" if tokenizer_files else "fail",
        "files": [path.name for path in tokenizer_files],
        "structural_digest": tokenizer_digest.hexdigest() if tokenizer_files else None,
    }
    return model_check, tokenizer_check


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


def _check_privacy(root: Path) -> dict[str, Any]:
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
    return {"status": "ok" if not problems else "fail", "problems": problems}


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


def inspect_bridge_bundle(root: str | Path, *, require_verl: bool = False) -> dict[str, Any]:
    """Return the complete bridge diagnosis; do not execute distributed code."""
    bundle = Path(root)
    target = _check_requirements(bundle)
    model, tokenizer = _check_model(bundle)
    parquet = _check_parquet(bundle)
    config = _check_config(bundle)
    reward = _check_reward(bundle)
    hashes = _check_hashes(bundle)
    privacy = _check_privacy(bundle)
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
