"""Strict, portable community hardware records derived from completed runs."""

from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from miniverl.utils.privacy import portable_text
from miniverl.utils.runs import read_json, read_jsonl, utc_now, write_json_atomic

__all__ = [
    "HardwareRecord",
    "build_hardware_record",
    "load_hardware_record",
    "validate_hardware_record",
    "write_hardware_record",
]

_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class EvidenceValue(_StrictModel):
    """One value with an explicit evidence state and source."""

    status: Literal["measured", "estimated", "unknown"]
    value: float | int | str | bool | None = None
    unit: str | None = None
    source: str

    @model_validator(mode="after")
    def _status_matches_value(self) -> EvidenceValue:
        if self.status in {"measured", "estimated"} and self.value is None:
            raise ValueError(f"{self.status} evidence requires a value")
        if self.status == "unknown" and self.value is not None:
            raise ValueError("unknown evidence must not invent a value")
        return self


class ProfileEvidence(_StrictModel):
    name: str
    identity_digest: str = Field(pattern=_SHA256_PATTERN)
    upstream_tag: str
    upstream_commit: str = Field(pattern=r"^[0-9a-f]{40}$")


class RunEvidence(_StrictModel):
    run_id: str
    status: Literal["completed"]
    manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    plan_digest: str = Field(pattern=_SHA256_PATTERN)
    resolved_config_digest: str = Field(pattern=_SHA256_PATTERN)


class HardwareEvidence(_StrictModel):
    measurement_status: Literal["measured", "estimated", "unknown"]
    gpu_name: str
    gpu_count: int = Field(ge=1)
    vram_gib: float = Field(gt=0)
    driver: EvidenceValue
    cuda_runtime: str
    torch: str
    os: str
    python: str


class ModelEvidence(_StrictModel):
    role: Literal["student", "teacher"]
    model_id: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    quantization: Literal["none", "nf4", "int8"]
    dtype: str


class BoundsEvidence(_StrictModel):
    prompt_tokens: int = Field(ge=1)
    response_tokens: int = Field(ge=1)


class TargetEvidence(_StrictModel):
    mode: str
    top_k: int | None = Field(default=None, ge=1)
    estimator: str | None = None


class BatchingEvidence(_StrictModel):
    logical: int = Field(ge=1)
    rollout_physical: int = Field(ge=1)
    teacher_score_physical: int = Field(ge=1)
    update_trajectory_physical: int = Field(ge=1)


class TimingEvidence(_StrictModel):
    rollout_seconds: EvidenceValue
    teacher_scoring_seconds: EvidenceValue
    actor_update_seconds: EvidenceValue


class MeasurementsEvidence(_StrictModel):
    peak_allocated_gib: EvidenceValue
    peak_reserved_gib: EvidenceValue
    phase_medians: TimingEvidence


class ResumeEvidence(_StrictModel):
    status: Literal["measured", "estimated", "unknown"]
    outcome: Literal["resumed", "not_exercised", "unknown"]
    checkpoint_digest: str | None = Field(default=None, pattern=_SHA256_PATTERN)


class ArtifactEvidence(_StrictModel):
    name: str
    sha256: str = Field(pattern=_SHA256_PATTERN)
    bytes: int = Field(ge=0)


class HardwareRecord(_StrictModel):
    """Version-1 portable record; validation does not confer publication trust."""

    schema_version: Literal[1] = 1
    kind: Literal["miniverl_hardware_record"] = "miniverl_hardware_record"
    created_at: str
    record_classification: Literal["maintainer_measured", "community_submitted", "not_measured"]
    review_status: Literal["unreviewed", "maintainer_validated"] = "unreviewed"
    miniverl_version: str
    profile: ProfileEvidence
    run: RunEvidence
    hardware: HardwareEvidence
    models: list[ModelEvidence] = Field(min_length=2, max_length=2)
    bounds: BoundsEvidence
    target: TargetEvidence
    batching: BatchingEvidence
    optimizer_updates: EvidenceValue
    measurements: MeasurementsEvidence
    resume: ResumeEvidence
    artifacts: list[ArtifactEvidence] = Field(min_length=1)
    scientific_scope: dict[str, bool]
    consent_to_publish: bool = False

    @model_validator(mode="after")
    def _trust_and_roles_are_closed(self) -> HardwareRecord:
        if [model.role for model in self.models] != ["student", "teacher"]:
            raise ValueError("models must contain student then teacher")
        if (
            self.record_classification == "maintainer_measured"
            and self.review_status != "maintainer_validated"
        ):
            raise ValueError("maintainer_measured requires maintainer_validated review status")
        if (
            self.review_status == "maintainer_validated"
            and self.record_classification != "maintainer_measured"
        ):
            raise ValueError("maintainer_validated records must be maintainer_measured")
        if self.review_status == "maintainer_validated" and not self.consent_to_publish:
            raise ValueError("maintainer validation for publication requires consent")
        if self.record_classification == "not_measured":
            raise ValueError("run-derived hardware records cannot be labelled not_measured")
        return self


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _measured(value: float | int, unit: str, source: str) -> EvidenceValue:
    return EvidenceValue(status="measured", value=value, unit=unit, source=source)


def _median(rows: list[dict[str, Any]], field: str) -> EvidenceValue:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    if not values:
        return EvidenceValue(status="unknown", source="metrics.jsonl")
    return _measured(round(float(statistics.median(values)), 4), "seconds", "metrics.jsonl")


def _artifact_records(run: Path) -> list[ArtifactEvidence]:
    candidates = {
        "checkpoint_adapter": run / "checkpoints/final/adapter.safetensors",
        "checkpoint_optimizer": run / "checkpoints/final/optimizer.safetensors",
        "checkpoint_state": run / "checkpoints/final/state.json",
        "trajectories": run / "trajectories.jsonl",
        "peft_adapter": run / "final-peft-adapter/adapter_model.safetensors",
    }
    return [
        ArtifactEvidence(name=name, sha256=_sha256(path), bytes=path.stat().st_size)
        for name, path in candidates.items()
        if path.is_file()
    ]


def build_hardware_record(
    run: str | Path,
    *,
    classification: Literal["maintainer_measured", "community_submitted"] = ("community_submitted"),
    review_status: Literal["unreviewed", "maintainer_validated"] = "unreviewed",
    consent_to_publish: bool = False,
) -> HardwareRecord:
    """Derive a portable candidate record without importing torch or uploading it."""
    root = Path(run).resolve(strict=True)
    manifest_path = root / "manifest.json"
    config_path = root / "config.validated.yaml"
    plan_path = root / "local-execution-plan.json"
    manifest = read_json(manifest_path)
    plan = read_json(plan_path)
    if manifest.get("status") != "completed":
        raise ValueError("hardware record requires a completed run manifest")
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot read validated run config: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError("validated run config must contain a mapping")
    profile = manifest.get("profile_identity") or {}
    gpu = manifest.get("gpu") or {}
    models = manifest.get("models") or {}
    objective = manifest.get("objective") or {}
    source = config.get("source") or {}
    rollout = config.get("rollout") or {}
    train = config.get("train") or {}
    loss_config = config.get("loss") or {}
    plan_batching = (plan.get("system_plan") or {}).get("batching") or {}
    cycle_rows = [
        row for row in read_jsonl(root / "metrics.jsonl") if row.get("phase") == "opd_cycle"
    ]
    update_rows = [row for row in read_jsonl(root / "metrics.jsonl") if row.get("phase") == "opd"]
    memory_rows = [
        row.get("memory") or {} for row in [*cycle_rows, *update_rows] if row.get("memory")
    ]
    cuda_status = (manifest.get("measurement_status") or {}).get("cuda_metrics")
    if cuda_status != "measured":
        raise ValueError("hardware record requires measured CUDA metrics")
    artifacts = _artifact_records(root)
    if not artifacts:
        raise ValueError("hardware record requires checksummed run artifacts")
    checkpoint = manifest.get("final_checkpoint") or {}
    resumed = manifest.get("resumed_from") is not None
    driver = gpu.get("driver_version")
    record = HardwareRecord(
        created_at=utc_now(),
        record_classification=classification,
        review_status=review_status,
        miniverl_version=str(manifest["miniverl_version"]),
        profile=ProfileEvidence(
            name=str(profile["profile_name"]),
            identity_digest=str(profile["digest"]),
            upstream_tag=str(profile["upstream_tag"]),
            upstream_commit=str(profile["upstream_commit"]),
        ),
        run=RunEvidence(
            run_id=str(manifest["run_id"]),
            status="completed",
            manifest_sha256=_sha256(manifest_path),
            plan_digest=str(manifest["execution_plan_digest"]),
            resolved_config_digest=str(manifest["resolved_config_digest"]),
        ),
        hardware=HardwareEvidence(
            measurement_status="measured",
            gpu_name=str(gpu["name"]),
            gpu_count=int(gpu["device_count"]),
            vram_gib=float(gpu["total_memory_gib"]),
            driver=(
                EvidenceValue(status="unknown", source="manifest.json")
                if driver is None
                else EvidenceValue(status="measured", value=str(driver), source="manifest.json")
            ),
            cuda_runtime=str(gpu["torch_cuda_version"]),
            torch=str(gpu["torch_version"]),
            os=str(manifest["platform"]),
            python=str(manifest["python_version"]),
        ),
        models=[
            ModelEvidence(
                role=role,
                model_id=str(models[role]["model_id"]),
                revision=str(models[role]["revision"]),
                quantization=str(models[role]["quantization"]),
                dtype=str(models[role]["capabilities"]["dtype"]),
            )
            for role in ("student", "teacher")
        ],
        bounds=BoundsEvidence(
            prompt_tokens=int(source["max_prompt_length"]),
            response_tokens=int(source["max_response_length"]),
        ),
        target=TargetEvidence(
            mode=str(objective["loss_mode"]),
            top_k=(
                None
                if objective["loss_mode"] == "verl_pg_k1"
                else int(objective["top_k"])
                if objective.get("top_k") is not None
                else None
            ),
            estimator=loss_config.get("estimator_implementation_version"),
        ),
        batching=BatchingEvidence(
            logical=int(train["rollouts_per_cycle"]),
            rollout_physical=int(rollout["prompt_batch_size"]),
            teacher_score_physical=int(plan_batching["teacher_score"]),
            update_trajectory_physical=int(train["trajectory_batch_size"]),
        ),
        optimizer_updates=_measured(
            int(manifest["actual_optimizer_updates"]), "updates", "manifest.json"
        ),
        measurements=MeasurementsEvidence(
            peak_allocated_gib=_measured(
                max(float(row.get("peak_allocated_gib") or 0.0) for row in memory_rows),
                "GiB",
                "metrics.jsonl",
            ),
            peak_reserved_gib=_measured(
                max(float(row.get("peak_reserved_gib") or 0.0) for row in memory_rows),
                "GiB",
                "metrics.jsonl",
            ),
            phase_medians=TimingEvidence(
                rollout_seconds=_median(cycle_rows, "rollout_seconds"),
                teacher_scoring_seconds=_median(cycle_rows, "teacher_scoring_seconds"),
                actor_update_seconds=_median(update_rows, "seconds"),
            ),
        ),
        resume=ResumeEvidence(
            status="measured" if resumed else "unknown",
            outcome="resumed" if resumed else "not_exercised",
            checkpoint_digest=(str(checkpoint.get("digest")) if resumed else None),
        ),
        artifacts=artifacts,
        scientific_scope={
            "runtime_correctness_only": True,
            "task_quality_evaluated": False,
            "alignment_quality_evaluated": False,
            "distributed_execution_tested": False,
        },
        consent_to_publish=consent_to_publish,
    )
    serialized = json.dumps(record.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
    if portable_text(serialized) != serialized:
        raise ValueError("generated hardware record contains private or non-portable text")
    return record


def load_hardware_record(path: str | Path) -> HardwareRecord:
    """Read and validate one standalone record."""
    return HardwareRecord.model_validate(read_json(path))


def validate_hardware_record(path: str | Path) -> list[str]:
    """Return validation problems; an empty list means schema-valid, not trusted."""
    try:
        payload = read_json(path)
        HardwareRecord.model_validate(payload)
    except Exception as exc:
        return [f"schema: {exc}"]
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    if portable_text(serialized) != serialized:
        return ["privacy: record contains a local path, credential, or environment reference"]
    return []


def write_hardware_record(record: HardwareRecord, path: str | Path) -> Path:
    """Publish one record atomically; no network operation is performed."""
    return write_json_atomic(path, record.model_dump(mode="json"))
