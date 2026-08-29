"""Torch-free validation for exact-commit single-GPU qualification evidence."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from miniverl.utils.privacy import portable_text

__all__ = [
    "GPUQualification",
    "qualification_json_schema",
    "sha256_file",
    "validate_qualification_file",
    "validate_qualification_payload",
]

_SHA256 = r"^[0-9a-f]{64}$"
_SHA1 = r"^[0-9a-f]{40}$"
_PRIVATE_PATH = re.compile(r"(?i)(?:[a-z]:\\users\\|/home/|/users/|onedrive)")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class WheelEvidence(_Strict):
    filename: str = Field(min_length=1, pattern=r"^[^/\\]+\.whl$")
    sha256: str = Field(pattern=_SHA256)


class CandidateBinding(_Strict):
    manifest_sha256: str = Field(pattern=_SHA256)
    artifact_name: Literal["candidate-distributions"]
    workflow_repository: str | None = None
    workflow_path: str | None = None
    workflow_run_id: int | None = Field(default=None, gt=0)
    workflow_run_attempt: int | None = Field(default=None, gt=0)
    installed_from_candidate: Literal[True]
    import_origin_verified: Literal[True]
    cli_origin_verified: Literal[True]
    import_origin: Literal["qualification_venv_site_packages"]

    @model_validator(mode="after")
    def _workflow_binding_is_complete_or_local(self) -> CandidateBinding:
        values = (
            self.workflow_repository,
            self.workflow_path,
            self.workflow_run_id,
            self.workflow_run_attempt,
        )
        if any(value is None for value in values) and any(value is not None for value in values):
            raise ValueError("candidate workflow identity must be complete or entirely local")
        return self


class ProfileEvidence(_Strict):
    name: str = Field(min_length=1)
    identity_digest: str = Field(pattern=_SHA256)
    upstream_tag: str = Field(min_length=1)
    upstream_commit: str = Field(pattern=_SHA1)


class EnvironmentEvidence(_Strict):
    known_good_manifest_sha256: str = Field(pattern=_SHA256)
    gpu_name: str = Field(min_length=1)
    gpu_count: Literal[1]
    vram_gib: float = Field(gt=0)
    driver: str = Field(min_length=1)
    cuda_runtime: str = Field(min_length=1)
    python: str = Field(min_length=1)
    packages: dict[str, str]

    @model_validator(mode="after")
    def _required_packages_are_present(self) -> EnvironmentEvidence:
        required = {
            "torch",
            "transformers",
            "peft",
            "accelerate",
            "bitsandbytes",
            "numpy",
            "pyarrow",
            "safetensors",
        }
        missing = sorted(required - self.packages.keys())
        if missing:
            raise ValueError("environment packages missing: " + ", ".join(missing))
        if any(not value for value in self.packages.values()):
            raise ValueError("environment package versions must be non-empty")
        return self


class ModelEvidence(_Strict):
    role: Literal["actor", "teacher"]
    model_id: str = Field(min_length=1)
    revision: str = Field(pattern=_SHA1)


class ExecutionEvidence(_Strict):
    rollout_completed: Literal[True]
    teacher_scoring_completed: Literal[True]
    optimizer_updates: int = Field(ge=1)
    peft_adapter_exported: Literal[True]
    peft_adapter_reload_verified: Literal[True]
    cuda_allocated_before_bytes: int = Field(ge=0)
    cuda_allocated_after_teardown_bytes: int = Field(ge=0)
    cuda_teardown_tolerance_bytes: int = Field(ge=0)

    @model_validator(mode="after")
    def _teardown_returns_to_baseline(self) -> ExecutionEvidence:
        ceiling = self.cuda_allocated_before_bytes + self.cuda_teardown_tolerance_bytes
        if self.cuda_allocated_after_teardown_bytes > ceiling:
            raise ValueError(
                "CUDA teardown left live allocations above the measured baseline tolerance"
            )
        return self


class InputEvidence(_Strict):
    config_sha256: str = Field(pattern=_SHA256)
    plan_sha256: str = Field(pattern=_SHA256)
    parquet_sha256: str = Field(pattern=_SHA256)


class ArtifactEvidence(_Strict):
    name: str = Field(min_length=1)
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=_SHA256)
    bytes: int = Field(ge=0)

    @model_validator(mode="after")
    def _portable_relative_path(self) -> ArtifactEvidence:
        path = PurePosixPath(self.path.replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts or self.path.startswith(("~", "\\")):
            raise ValueError("artifact path must be a portable relative path")
        if any(part in {"", "."} for part in path.parts):
            raise ValueError("artifact path must be normalized")
        return self


class CheckEvidence(_Strict):
    executed: list[str]
    skipped: list[str]
    not_applicable: list[str]

    @model_validator(mode="after")
    def _states_are_disjoint(self) -> CheckEvidence:
        groups = [set(self.executed), set(self.skipped), set(self.not_applicable)]
        if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
            raise ValueError("check states must be disjoint")
        required = {
            "wheel_install",
            "doctor",
            "plan",
            "run_dry_run",
            "real_actor_teacher_update",
            "peft_reload",
            "cuda_teardown",
        }
        missing = sorted(required - groups[0])
        if missing:
            raise ValueError(
                "required release-smoke checks were not executed: " + ", ".join(missing)
            )
        return self


class ScientificScope(_Strict):
    runtime_correctness_only: Literal[True]
    task_quality_evaluated: Literal[False]
    alignment_quality_evaluated: Literal[False]
    distributed_execution_tested: Literal[False]
    other_hardware_measured: Literal[False]


class GPUQualification(_Strict):
    """Strict version-1 exact-commit qualification record."""

    schema_version: Literal[1] = 1
    kind: Literal["miniverl_gpu_qualification"] = "miniverl_gpu_qualification"
    level: Literal["release_smoke", "full_qualification"]
    status: Literal["passed"]
    measured_at: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
    source_commit: str = Field(pattern=_SHA1)
    miniverl_version: str = Field(min_length=1)
    wheel: WheelEvidence
    candidate: CandidateBinding
    profile: ProfileEvidence
    environment: EnvironmentEvidence
    models: list[ModelEvidence] = Field(min_length=2, max_length=2)
    execution: ExecutionEvidence
    inputs: InputEvidence
    artifacts: list[ArtifactEvidence] = Field(min_length=1)
    checks: CheckEvidence
    scientific_scope: ScientificScope

    @model_validator(mode="after")
    def _roles_and_gpu_are_exact(self) -> GPUQualification:
        if [model.role for model in self.models] != ["actor", "teacher"]:
            raise ValueError("models must contain actor then teacher")
        if self.level == "full_qualification":
            required = {
                "direct_gkd_resume_equivalence",
                "sampled_k1_resume_equivalence",
                "smollm2_resume_equivalence",
                "export_materialize_doctor",
            }
            missing = sorted(required - set(self.checks.executed))
            if missing:
                raise ValueError(
                    "full qualification checks were not executed: " + ", ".join(missing)
                )
            required_artifacts = {
                "release_smoke_record",
                "full_direct_result",
                "full_pg_k1_result",
                "full_smollm2_result",
            }
            missing_artifacts = sorted(
                required_artifacts - {artifact.name for artifact in self.artifacts}
            )
            if missing_artifacts:
                raise ValueError(
                    "full qualification evidence is missing: " + ", ".join(missing_artifacts)
                )
            version_parts = self.miniverl_version.split(".", 2)
            try:
                requires_v011 = (int(version_parts[0]), int(version_parts[1])) >= (0, 11)
            except (IndexError, ValueError):
                requires_v011 = False
            if requires_v011:
                v011_checks = {
                    "v011_hf_cached_direct_n1",
                    "v011_hf_cached_pg_n4",
                    "v011_rewarded_pg_n4",
                    "v011_exact_wheel_runtime_gate",
                    "v011_vllm_direct_gkd_n4_r256",
                    "v011_policy_refresh_cache_invalidation",
                    "v011_external_engine_teardown",
                }
                missing_v011_checks = sorted(v011_checks - set(self.checks.executed))
                v011_artifacts = {
                    "full_v011_profiles_result",
                    "full_hf_cached_runtime_result",
                    "full_vllm_runtime_result",
                }
                missing_v011_artifacts = sorted(
                    v011_artifacts - {artifact.name for artifact in self.artifacts}
                )
                if missing_v011_checks or missing_v011_artifacts:
                    details = missing_v011_checks + missing_v011_artifacts
                    raise ValueError(
                        "v0.11 full qualification evidence is missing: " + ", ".join(details)
                    )
        return self


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def qualification_json_schema() -> dict[str, Any]:
    return GPUQualification.model_json_schema()


def _finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    return True


def validate_qualification_payload(
    payload: Any,
    *,
    artifact_root: str | Path | None = None,
    expected_commit: str | None = None,
    expected_wheel_sha256: str | None = None,
    expected_candidate_manifest_sha256: str | None = None,
    expected_known_good_sha256: str | None = None,
    required_gpu_name: str | None = None,
) -> list[str]:
    """Validate evidence without importing torch; empty means schema-valid and bound."""
    if not _finite(payload):
        return ["schema: qualification JSON contains NaN or infinity"]
    try:
        record = GPUQualification.model_validate(payload)
    except Exception as exc:
        return [f"schema: {exc}"]
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False)
    if portable_text(serialized) != serialized or _PRIVATE_PATH.search(serialized):
        return [
            "privacy: qualification contains a local path, credential, or environment reference"
        ]
    problems: list[str] = []
    if expected_commit is not None and record.source_commit != expected_commit:
        problems.append(
            f"binding: source commit {record.source_commit} does not match {expected_commit}"
        )
    if expected_wheel_sha256 is not None and record.wheel.sha256 != expected_wheel_sha256:
        problems.append("binding: wheel checksum does not match the expected release wheel")
    if (
        expected_candidate_manifest_sha256 is not None
        and record.candidate.manifest_sha256 != expected_candidate_manifest_sha256
    ):
        problems.append("binding: candidate manifest checksum does not match")
    if (
        expected_known_good_sha256 is not None
        and record.environment.known_good_manifest_sha256 != expected_known_good_sha256
    ):
        problems.append("binding: known-good environment checksum does not match")
    if required_gpu_name is not None and record.environment.gpu_name != required_gpu_name:
        problems.append(
            f"binding: GPU {record.environment.gpu_name!r} is not required {required_gpu_name!r}"
        )
    if artifact_root is None:
        return problems
    root = Path(artifact_root).resolve(strict=True)
    candidates = [(record.wheel.filename, record.wheel.sha256, None)] + [
        (item.path, item.sha256, item.bytes) for item in record.artifacts
    ]
    referenced = {Path(relative).as_posix() for relative, _, _ in candidates}
    for relative, expected, expected_bytes in candidates:
        path = root / relative
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(root)
        except (OSError, ValueError):
            problems.append(f"artifact: unsafe or missing path {relative!r}")
            continue
        if resolved.is_symlink() or not resolved.is_file():
            problems.append(f"artifact: {relative!r} must be a regular file")
            continue
        actual_bytes = resolved.stat().st_size
        if expected_bytes is not None and actual_bytes != expected_bytes:
            problems.append(
                f"artifact size mismatch for {relative}: "
                f"expected {expected_bytes}, got {actual_bytes}"
            )
        actual = sha256_file(resolved)
        if actual != expected:
            problems.append(
                f"artifact checksum mismatch for {relative}: expected {expected}, got {actual}"
            )
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    unreferenced = sorted(actual_files - referenced - {"qualification.json"})
    if unreferenced:
        problems.append("artifact: unreferenced files are not allowed: " + ", ".join(unreferenced))
    return problems


def validate_qualification_file(
    path: str | Path,
    **kwargs: Any,
) -> list[str]:
    target = Path(path)
    try:
        payload = json.loads(
            target.read_text(encoding="utf-8"),
            parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        return [f"json: {exc}"]
    return validate_qualification_payload(payload, artifact_root=target.parent, **kwargs)
