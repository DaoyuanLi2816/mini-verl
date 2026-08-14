"""Transactional materialization of a pure-OPD scale-out bundle.

The exporter deliberately publishes identity-only base-model records.  This
module turns those records into byte-bound local snapshots, validates the
documented pinned verl handoff without launching distributed execution, and
only then replaces the fail-closed launch template with ``launch.sh``.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import yaml

from miniverl import __version__
from miniverl.bridge.contract import VERL_COMMIT
from miniverl.bridge.opd_pg_v08 import VERL_OPD_PG_K1_V08_PROFILE
from miniverl.bridge.opd_v08 import VERL_OPD_V08_PROFILE
from miniverl.bridge.preflight import preflight_bundle_tree
from miniverl.bridge.safetensors_check import inspect_safetensors
from miniverl.errors import ConfigError
from miniverl.utils.runs import read_json, write_json, write_text

__all__ = ["materialize_verl_bundle"]

_REVISION_LENGTH = 40
_MAX_SNAPSHOT_FILES = 100_000
_MAX_SNAPSHOT_BYTES = 16 * 1024 * 1024 * 1024
_IGNORED_TOP_LEVEL = {".cache", ".git"}


def _validate_vocab_domain(*, model_vocab_size: int, tokenizer_max_token_id: int) -> None:
    """Accept padded model vocabularies while rejecting unreachable tokenizer IDs."""
    if model_vocab_size <= tokenizer_max_token_id:
        raise ValueError("causal-LM logits do not cover the tokenizer ID domain")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _immutable_revision(value: Any, *, role: str) -> str:
    revision = str(value or "")
    if len(revision) != _REVISION_LENGTH or any(
        char not in "0123456789abcdef" for char in revision
    ):
        raise ConfigError(
            f"{role} identity does not use an immutable 40-character revision",
            hint="export again from a run pinned to an exact model commit",
        )
    return revision


def _snapshot_manifest(path: Path) -> dict[str, Any]:
    manifest = path / "miniverl-snapshot.json"
    if not manifest.is_file():
        return {}
    try:
        payload = read_json(manifest)
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot read local snapshot identity manifest: {exc}") from exc
    return payload if isinstance(payload, dict) else {}


def _validate_local_revision(path: Path, *, model_id: str, revision: str, downloaded: bool) -> None:
    if downloaded:
        return
    manifest = _snapshot_manifest(path)
    manifest_files = manifest.get("files")
    manifest_files_dict: dict[str, Any] = manifest_files if isinstance(manifest_files, dict) else {}
    manifest_matches = (
        manifest.get("model_id") == model_id
        and manifest.get("revision") == revision
        and bool(manifest_files_dict)
    )
    # Hugging Face's immutable cache layout ends in snapshots/<commit>.  A
    # miniVERL manifest is the portable equivalent for copied local snapshots.
    expected_cache_folder = "models--" + model_id.replace("/", "--")
    cache_path_matches = (
        path.name == revision
        and path.parent.name == "snapshots"
        and path.parents[1].name == expected_cache_folder
    )
    if not manifest_matches and not cache_path_matches:
        raise ConfigError(
            f"local snapshot is not bound to the expected immutable revision {revision}",
            hint=(
                "pass the Hugging Face cache snapshots/<commit> directory, use --download, "
                "or provide miniverl-snapshot.json with the exact model_id and revision"
            ),
        )
    if manifest_matches:
        actual_files = {
            item.relative_to(path).as_posix()
            for item in path.rglob("*")
            if item.is_file()
            and item.name != "miniverl-snapshot.json"
            and not (
                item.relative_to(path).parts
                and item.relative_to(path).parts[0] in _IGNORED_TOP_LEVEL
            )
        }
        if set(manifest_files_dict) != actual_files:
            raise ConfigError("local snapshot manifest does not enumerate every snapshot file")
        for relative, expected in sorted(manifest_files_dict.items()):
            if not isinstance(relative, str) or not isinstance(expected, str):
                raise ConfigError("local snapshot manifest has an invalid file hash entry")
            relative_path = Path(relative)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ConfigError("local snapshot manifest path escapes its snapshot")
            candidate = path / relative_path
            if not candidate.is_file() or _sha256(candidate) != expected:
                raise ConfigError(f"local snapshot manifest hash mismatch: {relative}")


def _snapshot_model_files(path: Path) -> list[Path]:
    direct = [path / "model.safetensors", path / "pytorch_model.bin"]
    files = [candidate for candidate in direct if candidate.is_file()]
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = path / index_name
        if not index_path.is_file():
            continue
        try:
            index = read_json(index_path)
            names = sorted(set(index["weight_map"].values()))
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ConfigError(f"invalid model shard index {index_path.name}: {exc}") from exc
        unsafe = [name for name in names if Path(name).is_absolute() or ".." in Path(name).parts]
        if unsafe:
            raise ConfigError(f"model shard index contains unsafe paths: {unsafe}")
        missing = [name for name in names if not (path / name).is_file()]
        if missing:
            raise ConfigError(f"model shard index references missing files: {missing}")
        files.extend(path / name for name in names)
    return sorted(set(files))


def _validate_snapshot_payload(path: Path, *, role: str) -> dict[str, Any]:
    tree = preflight_bundle_tree(
        path,
        max_files=_MAX_SNAPSHOT_FILES,
        max_nominal_bytes=_MAX_SNAPSHOT_BYTES,
        max_depth=64,
    )
    if tree["status"] != "ok":
        reasons = ", ".join(
            f"{item['path']}: {item['reason']}" for item in tree.get("rejections", [])
        )
        raise ConfigError(
            f"{role} snapshot contains a symlink, reparse point, or unsafe entry: {reasons}"
        )
    config_path = path / "config.json"
    try:
        config = read_json(config_path)
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"{role} snapshot has no valid config.json: {exc}") from exc
    if not isinstance(config, dict):
        raise ConfigError(f"{role} snapshot config.json must contain an object")
    model_files = _snapshot_model_files(path)
    if not model_files:
        raise ConfigError(f"{role} snapshot has no model weights or complete shard index")
    safetensors = [item for item in model_files if item.suffix == ".safetensors"]
    for item in safetensors:
        check = inspect_safetensors(item)
        if check["status"] != "ok":
            raise ConfigError(f"{role} snapshot has invalid {item.name}: {check['detail']}")
    tokenizer_config = path / "tokenizer_config.json"
    tokenizer_vocab = (
        path / "tokenizer.json",
        path / "tokenizer.model",
        path / "vocab.json",
        path / "vocab.txt",
    )
    if not tokenizer_config.is_file() or not any(item.is_file() for item in tokenizer_vocab):
        raise ConfigError(f"{role} snapshot is missing tokenizer config or vocabulary files")
    return {
        "tree": tree,
        "model_files": [item.name for item in model_files],
        "config_sha256": _sha256(config_path),
        "weight_format": "safetensors" if safetensors else "pytorch_bin",
    }


def _copy_snapshot_tree(source: Path, destination: Path, *, bundle_prefix: str) -> dict[str, str]:
    destination.mkdir(parents=True)
    anchor = source.resolve(strict=True)
    files: dict[str, str] = {}
    for item in sorted(source.rglob("*")):
        relative = item.relative_to(source)
        if relative.parts and relative.parts[0] in _IGNORED_TOP_LEVEL:
            continue
        entry_stat = item.lstat()
        if item.is_symlink() or getattr(entry_stat, "st_file_attributes", 0) & 0x400:
            raise ConfigError(f"snapshot changed to a symlink or reparse point: {relative}")
        try:
            item.resolve(strict=True).relative_to(anchor)
        except (OSError, ValueError) as exc:
            raise ConfigError(f"snapshot entry escapes during copy: {relative}") from exc
        target = destination / relative
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target)
        files[f"{bundle_prefix}/{relative.as_posix()}"] = _sha256(target)
    return files


def _tree_file_hashes(source: Path, *, prefix: str) -> dict[str, str]:
    files: dict[str, str] = {}
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if relative.parts and relative.parts[0] in _IGNORED_TOP_LEVEL:
            continue
        if path.is_file():
            files[f"{prefix}/{relative.as_posix()}"] = _sha256(path)
    return files


def _copy_cached_snapshot_to_regular(source: Path, destination: Path) -> None:
    """Dereference only links that stay inside one Hugging Face repository cache."""
    try:
        repository_cache = source.parents[1].resolve(strict=True)
    except (IndexError, OSError) as exc:
        raise ConfigError("cached snapshot does not have the expected repository layout") from exc
    destination.mkdir(parents=True)
    for item in sorted(source.rglob("*")):
        relative = item.relative_to(source)
        target = destination / relative
        try:
            attributes = getattr(item.lstat(), "st_file_attributes", 0)
        except OSError as exc:
            raise ConfigError(f"cached snapshot entry cannot be inspected: {relative}") from exc
        if attributes & 0x400 and not item.is_symlink():
            raise ConfigError(f"cached snapshot contains a reparse point: {relative}")
        if item.is_dir():
            if item.is_symlink():
                raise ConfigError(f"cached snapshot contains a linked directory: {relative}")
            target.mkdir(parents=True, exist_ok=True)
            continue
        try:
            resolved = item.resolve(strict=True)
            resolved.relative_to(repository_cache)
        except (OSError, ValueError) as exc:
            raise ConfigError(f"cached snapshot link escapes its repository: {relative}") from exc
        if not resolved.is_file():
            raise ConfigError(f"cached snapshot entry is not a regular file: {relative}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(resolved, target)


def _download_snapshot(*, model_id: str, revision: str, destination: Path, offline: bool) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ConfigError(
            "snapshot download requires huggingface_hub",
            hint="install miniverl[train] or pass exact local snapshot directories",
        ) from exc
    try:
        if offline:
            try:
                cached = Path(
                    snapshot_download(
                        repo_id=model_id,
                        revision=revision,
                        local_files_only=True,
                    )
                )
            except Exception:
                from huggingface_hub.constants import HF_HUB_CACHE
                from huggingface_hub.file_download import repo_folder_name

                cached = (
                    Path(HF_HUB_CACHE)
                    / repo_folder_name(repo_id=model_id, repo_type="model")
                    / "snapshots"
                    / revision
                )
                if not cached.is_dir():
                    raise
            _copy_cached_snapshot_to_regular(cached, destination)
            resolved = str(destination)
        else:
            resolved = snapshot_download(
                repo_id=model_id,
                revision=revision,
                local_dir=destination,
                local_files_only=False,
            )
    except Exception as exc:
        mode = "offline cache resolution" if offline else "snapshot download"
        raise ConfigError(f"{mode} failed for {model_id}@{revision}: {exc}") from exc
    return Path(resolved)


def _resolve_snapshot(
    supplied: Path | None,
    *,
    model_id: str,
    revision: str,
    download: bool,
    offline: bool,
    destination: Path,
    role: str,
) -> tuple[Path, bool]:
    if supplied is not None and download:
        raise ConfigError(f"choose either --{role}-snapshot or --download, not both")
    if supplied is None and not download:
        raise ConfigError(
            f"{role} snapshot is required",
            hint=f"pass --{role}-snapshot snapshots/{revision} or use --download",
        )
    if supplied is not None:
        return supplied.resolve(strict=True), False
    return _download_snapshot(
        model_id=model_id,
        revision=revision,
        destination=destination,
        offline=offline,
    ), True


def _merge_teacher_adapter(base: Path, adapter: Path, destination: Path) -> dict[str, Any]:
    try:
        import peft
        import torch
        import transformers
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ConfigError(
            "teacher adapter merge requires the training dependencies",
            hint="install miniverl[train] before using --merge-teacher-adapter",
        ) from exc
    model = AutoModelForCausalLM.from_pretrained(
        str(base), local_files_only=True, trust_remote_code=False, device_map="cpu"
    )
    merged = PeftModel.from_pretrained(model, str(adapter), is_trainable=False).merge_and_unload()
    destination.mkdir(parents=True)
    merged.save_pretrained(destination, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(
        str(base), local_files_only=True, trust_remote_code=False
    )
    tokenizer.save_pretrained(destination)
    for source in sorted(base.iterdir()):
        if source.is_file() and source.name.upper().startswith(("LICENSE", "NOTICE", "COPYING")):
            shutil.copy2(source, destination / source.name)
    del tokenizer, merged, model
    return {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
    }


def _validate_teacher_adapter(
    adapter: Path, *, teacher_id: str, teacher_revision: str, adapter_revision: str
) -> dict[str, Any]:
    tree = preflight_bundle_tree(adapter)
    if tree["status"] != "ok":
        raise ConfigError("teacher adapter tree failed structural preflight")
    try:
        config = read_json(adapter / "adapter_config.json")
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"teacher adapter config is invalid: {exc}") from exc
    if str(config.get("peft_type", "")).upper() != "LORA":
        raise ConfigError("teacher adapter is not a standard LoRA PEFT adapter")
    if config.get("base_model_name_or_path") != teacher_id:
        raise ConfigError("teacher adapter base model differs from the recorded teacher")
    if config.get("revision") != teacher_revision:
        raise ConfigError("teacher adapter base revision differs from the recorded teacher")
    payload = inspect_safetensors(adapter / "adapter_model.safetensors")
    if payload["status"] != "ok":
        raise ConfigError(f"teacher adapter payload is invalid: {payload['detail']}")
    return {
        "adapter_revision": adapter_revision,
        "config_sha256": _sha256(adapter / "adapter_config.json"),
        "payload_sha256": _sha256(adapter / "adapter_model.safetensors"),
    }


def _copy_teacher_adapter(source: Path, destination: Path) -> None:
    required = ("adapter_config.json", "adapter_model.safetensors")
    if not all((source / name).is_file() for name in required):
        raise ConfigError("downloaded teacher adapter is missing standard PEFT files")
    destination.mkdir(parents=True)
    for name in required:
        shutil.copy2(source / name, destination / name)
    for path in sorted(source.iterdir()):
        if path.is_file() and path.name.upper().startswith(("LICENSE", "NOTICE", "COPYING")):
            shutil.copy2(path, destination / path.name)


def _load_local_transformers_metadata(
    path: Path, *, role: str, adapter: Path | None = None
) -> dict[str, Any]:
    try:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ConfigError(
            "upstream materialization validation requires transformers",
            hint="install miniverl[train,bridge]",
        ) from exc
    try:
        config = AutoConfig.from_pretrained(
            str(path), local_files_only=True, trust_remote_code=False
        )
        tokenizer = AutoTokenizer.from_pretrained(
            str(path), local_files_only=True, trust_remote_code=False
        )
        from miniverl.models.tokenizers import tokenizer_structural_digest

        tokenizer_digest = tokenizer_structural_digest(tokenizer)
        model: Any = AutoModelForCausalLM.from_pretrained(
            str(path),
            local_files_only=True,
            trust_remote_code=False,
            low_cpu_mem_usage=True,
            device_map="cpu",
        )
        if adapter is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
        model.eval()
        encoded = tokenizer("materialization smoke", return_tensors="pt")
        with torch.no_grad():
            output = model(**encoded, use_cache=False)
        model_vocab_size = int(output.logits.shape[-1]) if output.logits.ndim == 3 else 0
        tokenizer_max_id = max(int(value) for value in tokenizer.get_vocab().values())
        if output.logits.ndim != 3:
            raise ValueError("causal-LM logits must have batch, sequence and vocabulary axes")
        _validate_vocab_domain(
            model_vocab_size=model_vocab_size, tokenizer_max_token_id=tokenizer_max_id
        )
    except Exception as exc:
        raise ConfigError(f"{role} local model/tokenizer smoke failed: {exc}") from exc
    metadata = {
        "config_class": type(config).__name__,
        "tokenizer_class": type(tokenizer).__name__,
        "vocab_size": len(tokenizer),
        "model_vocab_size": model_vocab_size,
        "tokenizer_max_token_id": tokenizer_max_id,
        "tokenizer_structural_digest_v2": tokenizer_digest,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "peft_adapter_loaded": adapter is not None,
        "tiny_forward": "passed on CPU without distributed execution",
    }
    del output, encoded, model, tokenizer, config
    gc.collect()
    return metadata


def _validate_upstream_bundle(root: Path) -> dict[str, Any]:
    report = read_json(root / "provenance/compatibility-report.json")
    profile = report.get("profile")
    try:
        distribution = importlib.metadata.distribution("verl")
    except importlib.metadata.PackageNotFoundError as exc:
        raise ConfigError(
            "the exact pinned verl distribution is required before launch publication",
            hint=f"install verl from commit {VERL_COMMIT} and retry",
        ) from exc
    direct_text = distribution.read_text("direct_url.json")
    try:
        direct = json.loads(direct_text) if direct_text else {}
    except json.JSONDecodeError as exc:
        raise ConfigError("installed verl has invalid direct_url.json") from exc
    actual_commit = ((direct or {}).get("vcs_info") or {}).get("commit_id")
    if actual_commit is None:
        raw_url = (direct or {}).get("url")
        if isinstance(raw_url, str) and urlparse(raw_url).scheme == "file":
            checkout = Path(unquote(urlparse(raw_url).path.lstrip("/")))
            try:
                actual_commit = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=checkout,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=10,
                ).stdout.strip()
            except (OSError, subprocess.SubprocessError):
                actual_commit = None
    if actual_commit != VERL_COMMIT:
        raise ConfigError(
            f"installed verl commit is {actual_commit or 'unverified'}, expected {VERL_COMMIT}"
        )
    try:
        from omegaconf import OmegaConf
    except ImportError as exc:
        raise ConfigError("pinned upstream validation requires omegaconf") from exc
    generated = Path(
        str(distribution.locate_file("verl/trainer/config/_generated_ppo_trainer.yaml"))
    )
    if not generated.is_file():
        raise ConfigError("installed pinned verl omits its generated PPO configuration")
    try:
        official = OmegaConf.load(generated)
        override = OmegaConf.load(root / "recipe/verl-opd-overrides.yaml")
        OmegaConf.set_struct(official, True)
        resolved = OmegaConf.merge(official, override)
        resolved_path = root / "recipe/verl-opd-resolved.yaml"
        OmegaConf.save(resolved, resolved_path)
    except Exception as exc:
        raise ConfigError(f"pinned upstream config merge failed: {exc}") from exc

    from miniverl.bridge.doctor import _check_model, _check_parquet

    adapter = _check_model(root, require_payload=True)
    parquet = _check_parquet(root, require_reward_model=False)
    if adapter["status"] != "ok":
        raise ConfigError(f"student PEFT validation failed: {adapter.get('detail')}")
    if parquet["status"] != "ok":
        raise ConfigError(f"Parquet validation failed: {parquet.get('detail')}")
    student = _load_local_transformers_metadata(
        root / "model/base", role="student", adapter=root / "model"
    )
    teacher = _load_local_transformers_metadata(root / "teacher/base", role="teacher")
    recipe = yaml.safe_load((root / "recipe/verl-opd-overrides.yaml").read_text(encoding="utf-8"))
    loss = recipe["distillation"]["distillation_loss"]
    semantic_checks: dict[str, Any]
    if profile == VERL_OPD_PG_K1_V08_PROFILE:
        semantic_checks = {
            "loss_mode": "k1",
            "use_policy_gradient": True,
            "policy_loss_mode": "vanilla",
            "use_task_rewards": False,
        }
        if "topk" in loss:
            raise ConfigError("sampled-k1 PG materialization forbids top-k targets")
        target_check = "sampled_k1_policy_gradient_contract"
    else:
        semantic_checks = {
            "loss_mode": "forward_kl_topk",
            "use_policy_gradient": False,
            "use_task_rewards": False,
        }
        topk = loss.get("topk")
        if isinstance(topk, bool) or not isinstance(topk, int) or topk < 1:
            raise ConfigError("pure-OPD top-k must be a positive integer")
        for role, metadata in (("student", student), ("teacher", teacher)):
            if topk > metadata["vocab_size"]:
                raise ConfigError(f"top-k {topk} exceeds the {role} tokenizer vocabulary")
        target_check = "topk_contract"
    for field, expected in semantic_checks.items():
        if loss.get(field) != expected:
            raise ConfigError(f"profile loss contract requires {field}={expected!r}")
    if student["tokenizer_structural_digest_v2"] != teacher["tokenizer_structural_digest_v2"]:
        raise ConfigError("student and teacher tokenizer structural identities differ")
    return {
        "status": "passed",
        "resolved_config": "recipe/verl-opd-resolved.yaml",
        "scope": (
            "exact verl config merge, Parquet footer, PEFT payload, local model config and "
            "tokenizer loads, sequential CPU model loads and tiny forwards, and "
            + (
                "the sampled-k1 policy-gradient contract; "
                if profile == VERL_OPD_PG_K1_V08_PROFILE
                else "top-k bounds; "
            )
            + "no distributed job was run"
        ),
        "checks": {
            "config_parse": "passed",
            "parquet_schema": "passed",
            "student_peft": "passed",
            "student_snapshot": student,
            "teacher_snapshot": teacher,
            target_check: "passed",
        },
    }


def _launch_script() -> str:
    return """#!/usr/bin/env bash
set -euo pipefail

BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BUNDLE_ROOT"
python -m verl.trainer.main_ppo \\
  --config-path "$BUNDLE_ROOT/recipe" \\
  --config-name verl-opd-resolved
"""


def _write_hashes(root: Path) -> None:
    checksum = root / "provenance/SHA256SUMS"
    lines = [
        f"{_sha256(path)}  {path.relative_to(root).as_posix()}"
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != checksum
    ]
    write_text(checksum, "\n".join(lines) + "\n")


def _replace_directory(source: Path, target: Path) -> None:
    backup = target.parent / f".{target.name}.{uuid.uuid4().hex}.backup"
    target.replace(backup)
    try:
        source.replace(target)
    except BaseException:
        backup.replace(target)
        raise
    shutil.rmtree(backup, ignore_errors=True)


def materialize_verl_bundle(
    bundle: str | Path,
    *,
    student_snapshot: str | Path | None = None,
    teacher_snapshot: str | Path | None = None,
    download: bool = False,
    offline: bool = False,
    merge_teacher_adapter: bool = False,
) -> dict[str, Any]:
    """Materialize exact base snapshots and publish a launchable bundle atomically."""
    root = Path(bundle).resolve(strict=True)
    tree = preflight_bundle_tree(root)
    if tree["status"] != "ok":
        raise ConfigError("bundle tree failed structural preflight")
    compatibility_path = root / "provenance/compatibility-report.json"
    try:
        report = read_json(compatibility_path)
        student_identity = read_json(root / "model/base-model.json")
        teacher_identity = read_json(root / "teacher/teacher-model.json")
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot read scale-out bundle identities: {exc}") from exc
    if report.get("profile") not in {VERL_OPD_V08_PROFILE, VERL_OPD_PG_K1_V08_PROFILE}:
        raise ConfigError("materialize supports only a registered pure-OPD v0.8 profile")
    student_id = str(student_identity.get("model_id") or "")
    teacher_id = str(teacher_identity.get("model_id") or "")
    student_revision = _immutable_revision(student_identity.get("revision"), role="student")
    teacher_revision = _immutable_revision(teacher_identity.get("revision"), role="teacher")
    teacher_adapter = teacher_identity.get("adapter") or {}
    adapter_required = bool(teacher_identity.get("upstream_materialization_required"))
    teacher_adapter_revision: str | None = None
    if adapter_required and not merge_teacher_adapter:
        raise ConfigError(
            "the recorded teacher adapter requires explicit --merge-teacher-adapter consent"
        )
    if adapter_required:
        teacher_adapter_revision = _immutable_revision(
            teacher_adapter.get("revision"), role="teacher adapter"
        )

    staging = root.parent / f".{root.name}.{uuid.uuid4().hex}.materializing"
    downloads = root.parent / f".{root.name}.{uuid.uuid4().hex}.downloads"
    downloads.mkdir()
    try:
        student_source, student_downloaded = _resolve_snapshot(
            Path(student_snapshot) if student_snapshot is not None else None,
            model_id=student_id,
            revision=student_revision,
            download=download,
            offline=offline,
            destination=downloads / "student",
            role="student",
        )
        teacher_source, teacher_downloaded = _resolve_snapshot(
            Path(teacher_snapshot) if teacher_snapshot is not None else None,
            model_id=teacher_id,
            revision=teacher_revision,
            download=download,
            offline=offline,
            destination=downloads / "teacher",
            role="teacher",
        )
        _validate_local_revision(
            student_source,
            model_id=student_id,
            revision=student_revision,
            downloaded=student_downloaded,
        )
        _validate_local_revision(
            teacher_source,
            model_id=teacher_id,
            revision=teacher_revision,
            downloaded=teacher_downloaded,
        )
        student_validation = _validate_snapshot_payload(student_source, role="student")
        teacher_validation = _validate_snapshot_payload(teacher_source, role="teacher")
        teacher_base_files = _tree_file_hashes(teacher_source, prefix="source-teacher-base")

        shutil.copytree(root, staging)
        student_destination = staging / "model/base"
        teacher_destination = staging / "teacher/base"
        student_files = _copy_snapshot_tree(
            student_source, student_destination, bundle_prefix="model/base"
        )
        merge_software: dict[str, Any] | None = None
        if adapter_required:
            adapter_path = staging / "teacher/adapter"
            if not adapter_path.is_dir():
                recorded_adapter = teacher_adapter.get("path")
                if not isinstance(recorded_adapter, str) or not recorded_adapter.strip():
                    raise ConfigError(
                        "the bundle has no teacher adapter payload or downloadable identity"
                    )
                adapter_source = _download_snapshot(
                    model_id=recorded_adapter,
                    revision=str(teacher_adapter_revision),
                    destination=downloads / "teacher-adapter",
                    offline=offline,
                )
                _copy_teacher_adapter(adapter_source, adapter_path)
            adapter_validation = _validate_teacher_adapter(
                adapter_path,
                teacher_id=teacher_id,
                teacher_revision=teacher_revision,
                adapter_revision=str(teacher_adapter_revision),
            )
            merge_software = _merge_teacher_adapter(
                teacher_source, adapter_path, teacher_destination
            )
            teacher_adapter_files = _tree_file_hashes(adapter_path, prefix="teacher/adapter")
            _validate_snapshot_payload(teacher_destination, role="merged teacher")
            teacher_files = {
                path.relative_to(staging).as_posix(): _sha256(path)
                for path in sorted(teacher_destination.rglob("*"))
                if path.is_file()
            }
        else:
            adapter_validation = None
            teacher_adapter_files = {}
            teacher_files = _copy_snapshot_tree(
                teacher_source, teacher_destination, bundle_prefix="teacher/base"
            )

        write_json(
            staging / "provenance/materialization-manifest.json",
            {
                "schema_version": 1,
                "miniverl_version": __version__,
                "student": {
                    "model_id": student_id,
                    "revision": student_revision,
                    "source": "downloaded_exact_revision"
                    if student_downloaded
                    else "local_exact_snapshot",
                    "validation": student_validation,
                    "files": student_files,
                },
                "teacher": {
                    "model_id": teacher_id,
                    "revision": teacher_revision,
                    "source": "downloaded_exact_revision"
                    if teacher_downloaded
                    else "local_exact_snapshot",
                    "base_validation": teacher_validation,
                    "base_files": teacher_base_files,
                    "adapter": teacher_adapter,
                    "adapter_validation": adapter_validation,
                    "adapter_files": teacher_adapter_files,
                    "adapter_merged": adapter_required,
                    "merge_software": merge_software,
                    "files": teacher_files,
                },
            },
        )
        upstream = _validate_upstream_bundle(staging)
        template = staging / "recipe/launch.template.sh"
        template.unlink(missing_ok=True)
        write_text(staging / "recipe/launch.sh", _launch_script())
        report.update(
            {
                "artifact_complete": True,
                "artifact_bundle_complete": True,
                "config_semantics_supported": True,
                "student_artifact_loadable": True,
                "teacher_artifact_loadable": True,
                "dataset_loadable": True,
                "upstream_parse_passed": True,
                "upstream_config_parse_passed": True,
                "upstream_tiny_smoke_passed": False,
                "model_data_load_smoke_passed": True,
                "launchable": True,
                "distributed_execution_tested": False,
                "distributed_execution_status": "not tested",
                "launch_blockers": [],
                "materialization": {
                    "manifest": "provenance/materialization-manifest.json",
                    "upstream_validation": upstream,
                    "launch_script": "recipe/launch.sh",
                },
            }
        )
        write_json(staging / "provenance/compatibility-report.json", report)
        _write_hashes(staging)
        final_tree = preflight_bundle_tree(staging)
        if final_tree["status"] != "ok":
            raise ConfigError("materialized bundle failed final structural preflight")
        _replace_directory(staging, root)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(downloads, ignore_errors=True)
    return report
