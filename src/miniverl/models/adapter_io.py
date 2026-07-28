"""Standard PEFT adapter export, validation and provenance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from miniverl import __version__
from miniverl.cache.store import sha256_file
from miniverl.config.models import (
    AdapterSource,
    MemoryStrategy,
    Quantization,
    RunConfig,
    TeacherAdapterConfig,
    TeacherModelConfig,
)
from miniverl.errors import BackendError, ConfigError
from miniverl.utils.env import collect_environment
from miniverl.utils.runs import RunPaths, canonical_json, read_json, write_json

__all__ = [
    "ADAPTER_MANIFEST",
    "validate_teacher_adapter",
    "export_adapter",
    "digest_tree",
]

ADAPTER_MANIFEST = "miniverl_adapter_manifest.json"
_ADAPTER_CONFIG = "adapter_config.json"
_ADAPTER_WEIGHTS = "adapter_model.safetensors"
_POLICY_EVAL_FIELDS = (
    "tag",
    "split",
    "tasks",
    "strict_task_success_rate",
    "lenient_diagnostic_success_rate",
    "valid_tool_call_rate",
    "tool_call_count",
    "final_answer_format_validity_rate",
    "avg_turns",
    "protocol_token_accuracy",
    "policy_competence_measurement_status",
)


def digest_tree(directory: str | Path) -> str:
    """Content digest over relative names, file digests and sizes."""
    root = Path(directory)
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        file_digest, size = sha256_file(path)
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_digest.encode("ascii"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _same_model_identity(expected: str, actual: str) -> bool:
    if expected == actual:
        return True
    expected_path = Path(expected)
    actual_path = Path(actual)
    if expected_path.exists() and actual_path.exists():
        return expected_path.resolve() == actual_path.resolve()
    return False


def _read_local_adapter(path: Path) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if not path.is_dir():
        raise BackendError(
            f"teacher adapter directory not found: {path}",
            hint="export it with `miniverl export-adapter`, or correct models.teacher.adapter.path",
        )
    missing = [name for name in (_ADAPTER_CONFIG, _ADAPTER_WEIGHTS) if not (path / name).is_file()]
    if missing:
        raise BackendError(
            f"teacher adapter {path} is incomplete (missing {', '.join(missing)})",
            hint="a standard PEFT adapter needs adapter_config.json and adapter_model.safetensors",
        )
    try:
        adapter_config = json.loads((path / _ADAPTER_CONFIG).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BackendError(f"cannot read {path / _ADAPTER_CONFIG}: {exc}") from exc
    manifest_path = path / ADAPTER_MANIFEST
    manifest = None
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BackendError(f"cannot read {manifest_path}: {exc}") from exc
    return adapter_config, manifest


def _read_hub_adapter(
    repo_id: str,
    revision: str,
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Path]]:
    """Download the reproducibility metadata and weights at one immutable revision."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - transformers installs it
        raise BackendError(
            "loading a Hub teacher adapter requires huggingface_hub",
            hint='pip install "miniverl[train]"',
        ) from exc

    downloaded: dict[str, Path] = {}
    for name in (_ADAPTER_CONFIG, _ADAPTER_WEIGHTS, ADAPTER_MANIFEST):
        try:
            downloaded[name] = Path(
                hf_hub_download(repo_id=repo_id, filename=name, revision=revision)
            )
        except Exception as exc:
            if name == ADAPTER_MANIFEST:
                continue
            raise BackendError(
                f"could not download teacher adapter file {name!r} from "
                f"{repo_id!r} at revision {revision!r}: {exc}"
            ) from exc
    try:
        adapter_config = json.loads(downloaded[_ADAPTER_CONFIG].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BackendError(f"cannot read downloaded {_ADAPTER_CONFIG}: {exc}") from exc
    manifest = None
    manifest_path = downloaded.get(ADAPTER_MANIFEST)
    if manifest_path is not None:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BackendError(f"cannot read downloaded {ADAPTER_MANIFEST}: {exc}") from exc
    return adapter_config, manifest, downloaded


def validate_teacher_adapter(
    adapter: TeacherAdapterConfig,
    teacher: TeacherModelConfig,
    *,
    tokenizer_fingerprint: str,
) -> dict[str, Any]:
    """Validate compatibility before the adapter can supervise training."""
    if adapter.source is AdapterSource.LOCAL:
        path = Path(adapter.path)
        adapter_config, manifest = _read_local_adapter(path)
        downloaded = {
            name: path / name for name in (_ADAPTER_CONFIG, _ADAPTER_WEIGHTS, ADAPTER_MANIFEST)
        }
    else:
        assert adapter.revision is not None
        adapter_config, manifest, downloaded = _read_hub_adapter(adapter.path, adapter.revision)

    peft_type = str(adapter_config.get("peft_type") or "").upper()
    if peft_type != "LORA":
        raise BackendError(
            f"teacher adapter type {peft_type or 'unknown'!r} is not supported",
            hint="export a standard PEFT LoRA adapter",
        )
    target_modules = adapter_config.get("target_modules")
    if not target_modules:
        raise BackendError("teacher LoRA adapter declares no target_modules")

    adapter_base = str(adapter_config.get("base_model_name_or_path") or "")
    if not adapter_base or not _same_model_identity(teacher.model_id, adapter_base):
        raise BackendError(
            f"teacher adapter base {adapter_base or 'unreported'!r} does not match "
            f"models.teacher.model_id={teacher.model_id!r}",
            hint="load the adapter with the same base model it was trained from",
        )

    manifest = manifest or {}
    expected_revision = adapter.base_model_revision or manifest.get("base_model_revision")
    if expected_revision and teacher.revision != expected_revision:
        raise BackendError(
            f"teacher base revision {teacher.revision!r} does not match adapter "
            f"revision {expected_revision!r}"
        )

    expected_tokenizer = adapter.tokenizer_fingerprint or manifest.get("tokenizer_fingerprint")
    if not expected_tokenizer:
        raise BackendError(
            "teacher adapter has no tokenizer fingerprint",
            hint="use a miniVERL-exported adapter, or set "
            "models.teacher.adapter.tokenizer_fingerprint explicitly",
        )
    if expected_tokenizer != tokenizer_fingerprint:
        raise BackendError(
            "teacher adapter tokenizer fingerprint does not match the run tokenizer",
            hint=f"expected {expected_tokenizer[:16]}..., got {tokenizer_fingerprint[:16]}...",
        )

    policy_evaluation = manifest.get("policy_evaluation")
    if adapter.require_policy_evaluation:
        if not isinstance(policy_evaluation, dict):
            raise BackendError(
                "teacher adapter has no recorded tool-policy evaluation",
                hint=(
                    "train with eval.enabled=true and export the adapter again; "
                    "SFT loss is not a teacher-competence measurement"
                ),
            )
        missing_metrics = [field for field in _POLICY_EVAL_FIELDS if field not in policy_evaluation]
        if missing_metrics:
            raise BackendError(
                "teacher adapter policy evaluation is incomplete: " + ", ".join(missing_metrics)
            )
        strict_success = policy_evaluation.get("strict_task_success_rate")
        if not isinstance(strict_success, (int, float)):
            raise BackendError(
                "teacher adapter policy evaluation has no numeric strict success rate"
            )
        if (
            adapter.minimum_strict_success_rate is not None
            and float(strict_success) < adapter.minimum_strict_success_rate
        ):
            raise BackendError(
                f"teacher adapter strict success {float(strict_success):.1%} is below "
                f"the prespecified gate {adapter.minimum_strict_success_rate:.1%}",
                hint="report the failed teacher evaluation; do not run the headline OPD arm",
            )

    checksums = manifest.get("checksums") or {}
    for name, expected in checksums.items():
        file_path = downloaded.get(str(name))
        if file_path is None or not file_path.is_file():
            raise BackendError(f"adapter checksum references missing file {name!r}")
        actual, _ = sha256_file(file_path)
        if actual != expected:
            raise BackendError(
                f"teacher adapter checksum mismatch for {name}: expected "
                f"{str(expected)[:16]}..., got {actual[:16]}..."
            )

    weights_digest, _ = sha256_file(downloaded[_ADAPTER_WEIGHTS])
    return {
        "source": adapter.source.value,
        "identity": (
            Path(adapter.path).name if adapter.source is AdapterSource.LOCAL else adapter.path
        ),
        "revision": adapter.revision,
        "base_model_id": teacher.model_id,
        "base_model_revision": teacher.revision,
        "tokenizer_fingerprint": expected_tokenizer,
        "peft_type": peft_type,
        "target_modules": sorted(str(name) for name in target_modules),
        "weights_sha256": weights_digest,
        "manifest_digest": (
            hashlib.sha256(canonical_json(manifest).encode("utf-8")).hexdigest()
            if manifest
            else None
        ),
        "policy_evaluation": policy_evaluation,
    }


def export_adapter(
    run_dir: str | Path,
    checkpoint: str | Path,
    out: str | Path,
    *,
    local_files_only: bool = False,
) -> tuple[dict[str, Any], Path]:
    """Export a miniVERL LoRA checkpoint using PEFT's serialization."""
    from miniverl.models.factory import build_student, build_tokenizer
    from miniverl.training.checkpoint import load_checkpoint

    paths = RunPaths.open(run_dir)
    config = RunConfig.from_yaml(paths.config_resolved)
    if config.models.backend.value != "hf" or not config.models.student.lora.enabled:
        raise ConfigError("export-adapter requires an HF run with models.student.lora.enabled=true")

    target = Path(out)
    if target.exists() and not target.is_dir():
        raise ConfigError(
            f"adapter output path is not a directory: {target}",
            hint="choose a new directory for the exported PEFT adapter",
        )
    if target.exists() and any(target.iterdir()):
        raise ConfigError(
            f"adapter output directory is not empty: {target}",
            hint="choose a new output directory so existing weights are not overwritten",
        )
    target.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_dir():
        raise ConfigError(f"checkpoint directory not found: {checkpoint_path}")

    export_config = config.model_copy(deep=True)
    export_config.models.device = "cpu"
    export_config.models.student.quantization = Quantization.NONE
    export_config.memory.strategy = MemoryStrategy.RESIDENT
    tokenizer = build_tokenizer(export_config, local_files_only=local_files_only)
    backend = build_student(
        export_config,
        tokenizer,
        device="cpu",
        local_files_only=local_files_only,
    )
    load_checkpoint(
        checkpoint_path,
        backend=backend,
        optimizer=None,
        device="cpu",
        include_optimizer=False,
        include_rng=False,
    )

    model = getattr(backend, "model", None)
    save_pretrained = getattr(model, "save_pretrained", None)
    if not callable(save_pretrained):
        raise BackendError("loaded student is not a PEFT model and cannot export an adapter")
    save_pretrained(target, safe_serialization=True)
    missing = [
        name for name in (_ADAPTER_CONFIG, _ADAPTER_WEIGHTS) if not (target / name).is_file()
    ]
    if missing:
        raise BackendError("PEFT export did not produce required files: " + ", ".join(missing))

    checksums = {
        name: sha256_file(target / name)[0] for name in (_ADAPTER_CONFIG, _ADAPTER_WEIGHTS)
    }
    env = collect_environment()
    policy_evaluation = None
    if paths.eval_json.is_file():
        summary = read_json(paths.eval_json)
        final_eval = summary.get("eval") if isinstance(summary, dict) else None
        if isinstance(final_eval, dict):
            policy_evaluation = {field: final_eval.get(field) for field in _POLICY_EVAL_FIELDS}
    manifest = {
        "schema_version": 1,
        "miniverl_version": __version__,
        "git_commit": env.get("git_commit"),
        "base_model_id": config.models.student.model_id,
        "base_model_revision": config.models.student.revision,
        "tokenizer_fingerprint": tokenizer.fingerprint,
        "source_run": paths.root.name,
        "source_checkpoint": checkpoint_path.name,
        "source_checkpoint_digest": digest_tree(checkpoint_path),
        "lora": config.models.student.lora.model_dump(mode="json"),
        "training_environment": env,
        "training_task": {
            "environment": config.environment.name,
            "difficulty": config.environment.difficulty,
            "protocol": "miniverl_tool_protocol_v1",
            "mode": config.run.mode.value,
        },
        "policy_evaluation": policy_evaluation,
        "checksums": checksums,
    }
    write_json(target / ADAPTER_MANIFEST, manifest)
    return manifest, target
