"""Immutable, data-bound execution plans for the pinned verl OPD profile."""

from __future__ import annotations

import hashlib
import json
import re
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from miniverl import __version__
from miniverl.bridge.opd_runtime import OPDSystemPlan, build_system_plan, compile_native_run_config
from miniverl.bridge.opd_v08 import CompiledLocalExecutionPlan
from miniverl.config.models import RunConfig, VerlParquetSourceConfig
from miniverl.data.verl_parquet import VerlParquetDataset
from miniverl.errors import ConfigError
from miniverl.utils.runs import canonical_json, write_json_atomic

__all__ = [
    "ImmutableOPDPlan",
    "build_immutable_opd_plan",
    "load_and_verify_immutable_opd_plan",
]

_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_BUILTIN = "builtin:qwen3-0.6b-1.7b-opd"


class ImmutableOPDPlan(BaseModel):
    """Complete local execution input whose digest excludes only its self-references."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    artifact_type: Literal["miniverl_immutable_opd_plan"] = "miniverl_immutable_opd_plan"
    miniverl_version: str
    profile: str
    profile_identity: dict[str, Any] = Field(default_factory=dict)
    pinned_verl: dict[str, str]
    source_config: dict[str, Any]
    overrides: list[dict[str, Any]]
    compatibility: list[dict[str, Any]]
    compatibility_acceptance: dict[str, Any]
    compiled_plan: dict[str, Any]
    system_plan: dict[str, Any]
    resolved_native_config: dict[str, Any]
    data: dict[str, Any]
    models: dict[str, Any]
    tokenizers: dict[str, Any]
    loss: dict[str, Any]
    execution_recommendations: dict[str, Any]
    hardware_probe: dict[str, Any] | None = None
    plan_digest: str


def _sha256(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _source_identity(source: str | Path) -> tuple[dict[str, Any], Path]:
    text = str(source)
    if text == _BUILTIN:
        resource = files("miniverl").joinpath("resources/qwen3_0_6b_1_7b_opd.yaml")
        payload = resource.read_bytes()
        return (
            {
                "kind": "packaged_builtin",
                "locator": _BUILTIN,
                "path": None,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
            },
            Path.cwd(),
        )
    path = Path(source).resolve()
    if not path.is_file() or path.is_symlink():
        raise ConfigError(f"source config is not a regular file: {path}")
    digest, size = _sha256(path)
    return (
        {
            "kind": "resolved_yaml",
            "locator": str(source),
            "path": str(path),
            "sha256": digest,
            "bytes": size,
        },
        path.parent,
    )


def _revision(identity: str, revision: str | None, *, role: str) -> dict[str, Any]:
    if revision is None or _COMMIT.fullmatch(revision) is None:
        raise ConfigError(
            f"immutable plan requires a 40-hex commit revision for the {role} model",
            hint=f"resolve {identity} to an immutable Hub commit before planning",
        )
    return {"model_id": identity, "revision": revision, "revision_kind": "immutable_commit"}


def _resolve_data_paths(native: RunConfig, base: Path) -> None:
    if not isinstance(native.source, VerlParquetSourceConfig):
        raise ConfigError("immutable OPD plan requires a verl Parquet source")

    def resolved(values: list[str]) -> list[str]:
        return [
            str((path if path.is_absolute() else base / path).resolve())
            for path in map(Path, values)
        ]

    native.source.train_files = resolved(native.source.train_files)
    native.source.val_files = resolved(native.source.val_files)


def _data_identity(native: RunConfig) -> dict[str, Any]:
    assert isinstance(native.source, VerlParquetSourceConfig)
    manifest = VerlParquetDataset(native.source).inspect()
    records: list[dict[str, Any]] = []
    for split, names in (("train", native.source.train_files), ("val", native.source.val_files)):
        for name in names:
            path = Path(name)
            if not path.is_file() or path.is_symlink():
                raise ConfigError(f"{split} data is not a regular file: {path}")
            digest, size = _sha256(path)
            records.append(
                {"split": split, "path": str(path.resolve()), "sha256": digest, "bytes": size}
            )
    return {
        "manifest": {
            "rows": manifest.rows,
            "schema_digest": manifest.schema_digest,
            "content_digest": manifest.content_digest,
            "files": list(manifest.files),
        },
        "files": records,
    }


def _digest_payload(payload: dict[str, Any]) -> str:
    normalized = json.loads(json.dumps(payload))
    normalized.pop("plan_digest", None)
    run = normalized.get("resolved_native_config", {}).get("run")
    if isinstance(run, dict):
        run["execution_plan_digest"] = None
    return hashlib.sha256(canonical_json(normalized).encode("utf-8")).hexdigest()


def build_immutable_opd_plan(
    compiled: CompiledLocalExecutionPlan,
    *,
    source: str | Path,
    system_plan: OPDSystemPlan | None = None,
    rollout_backend: str | None = None,
) -> ImmutableOPDPlan:
    """Scan local inputs and build a deterministic, weight-free execution artifact."""
    acceptance = compiled.reinterpretation_acceptance
    if acceptance["required_fields"] and not acceptance["accepted"]:
        raise ConfigError(
            "cannot publish an execution plan with unaccepted local reinterpretations",
            hint="inspect the mappings, then pass --accept-local-reinterpretations",
        )
    source_identity, base = _source_identity(source)
    system = system_plan or build_system_plan(compiled)
    native = compile_native_run_config(
        compiled,
        system_plan=system,
        rollout_backend=rollout_backend,
    )
    _resolve_data_paths(native, base)
    student = _revision(
        native.models.student.model_id,
        native.models.student.revision,
        role="student",
    )
    teacher = _revision(
        native.models.teacher.model_id,
        native.models.teacher.revision,
        role="teacher",
    )
    from miniverl.bridge.profiles import get_profile

    profile_identity = get_profile(compiled.profile).identity.model_dump(mode="json")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "miniverl_immutable_opd_plan",
        "miniverl_version": __version__,
        "profile": compiled.profile,
        "profile_identity": profile_identity,
        "pinned_verl": compiled.upstream,
        "source_config": source_identity,
        "overrides": [item.model_dump(mode="json") for item in compiled.overrides],
        "compatibility": [item.model_dump(mode="json") for item in compiled.compatibility],
        "compatibility_acceptance": acceptance,
        "compiled_plan": compiled.model_dump(mode="json"),
        "system_plan": system.model_dump(mode="json"),
        "resolved_native_config": native.model_dump(mode="json"),
        "data": _data_identity(native),
        "models": {
            "student": student,
            "teacher": teacher,
            "teacher_adapter": compiled.source.miniverl.teacher_adapter.model_dump(mode="json"),
        },
        "tokenizers": {
            "status": "declared_not_loaded",
            "student": {"model_id": student["model_id"], "revision": student["revision"]},
            "teacher": {"model_id": teacher["model_id"], "revision": teacher["revision"]},
            "structural_identity": None,
            "behavioral_fingerprint": None,
        },
        "loss": system.loss,
        "execution_recommendations": {
            "local_execution": system.local_execution,
            "memory": system.memory,
            "batching": system.batching,
            "disk": system.disk,
            "time_to_first_update": system.time_to_first_update,
            "evidence_status": "estimated_or_unknown_not_measured",
        },
        "hardware_probe": None,
        "plan_digest": "0" * 64,
    }
    digest = _digest_payload(payload)
    payload["plan_digest"] = digest
    payload["resolved_native_config"]["run"]["execution_plan_digest"] = digest
    return ImmutableOPDPlan.model_validate(payload)


def _verify_version(plan: ImmutableOPDPlan) -> None:
    def release_line(value: str) -> tuple[str, str]:
        match = re.match(r"^(\d+)\.(\d+)", value)
        if match is None:
            raise ConfigError(f"invalid miniVERL version in execution plan: {value!r}")
        return match.group(1), match.group(2)

    if release_line(plan.miniverl_version) != release_line(__version__):
        raise ConfigError(
            f"execution plan targets miniVERL {plan.miniverl_version}, current is {__version__}",
            hint="rebuild the plan with this miniVERL minor release",
        )


def _verify_source(identity: dict[str, Any]) -> None:
    locator = identity.get("locator")
    if identity.get("kind") == "packaged_builtin":
        current, _ = _source_identity(str(locator))
    else:
        current, _ = _source_identity(Path(str(identity.get("path"))))
    if current["sha256"] != identity.get("sha256") or current["bytes"] != identity.get("bytes"):
        raise ConfigError("source config changed after the execution plan was written")


def load_and_verify_immutable_opd_plan(path: str | Path) -> tuple[ImmutableOPDPlan, RunConfig]:
    """Validate artifact, source, data and exact native config before model construction."""
    target = Path(path)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
        plan = ImmutableOPDPlan.model_validate(payload)
    except (OSError, UnicodeError, json.JSONDecodeError, ValidationError) as exc:
        raise ConfigError(f"cannot read immutable execution plan {target}: {exc}") from exc
    actual = _digest_payload(plan.model_dump(mode="json"))
    if actual != plan.plan_digest:
        raise ConfigError(
            f"execution plan digest mismatch: recorded {plan.plan_digest}, calculated {actual}",
            hint="do not edit plan.json; rebuild it from the source config",
        )
    _verify_version(plan)
    if plan.compatibility_acceptance.get(
        "required_fields"
    ) and not plan.compatibility_acceptance.get("accepted"):
        raise ConfigError("execution plan does not accept its required local reinterpretations")
    _verify_source(plan.source_config)
    for record in plan.data.get("files", []):
        data_path = Path(str(record.get("path")))
        if not data_path.is_file() or data_path.is_symlink():
            raise ConfigError(f"planned data file changed or disappeared: {data_path}")
        digest, size = _sha256(data_path)
        if digest != record.get("sha256") or size != record.get("bytes"):
            raise ConfigError(f"planned data file changed after planning: {data_path}")
    try:
        native = RunConfig.model_validate(plan.resolved_native_config)
    except ValidationError as exc:
        raise ConfigError(f"execution plan contains an invalid native config: {exc}") from exc
    observed_data = _data_identity(native)
    if observed_data != plan.data:
        raise ConfigError("planned data schema, rows, or content changed after planning")
    if native.run.execution_plan_digest != plan.plan_digest:
        raise ConfigError("native config is not bound to the execution plan digest")
    return plan, native


def write_immutable_opd_plan(path: str | Path, plan: ImmutableOPDPlan) -> None:
    """Atomically publish a canonical plan artifact."""
    write_json_atomic(Path(path), plan.model_dump(mode="json"))


def attach_hardware_probe(
    plan: ImmutableOPDPlan, hardware_probe: dict[str, Any]
) -> ImmutableOPDPlan:
    """Bind one measured probe to a new immutable plan digest."""
    payload = plan.model_dump(mode="json")
    bound_probe = json.loads(json.dumps(hardware_probe))
    # Cache path/reuse is invocation-local transport metadata. The measured
    # payload is identical whether freshly measured or loaded from its exact-
    # identity cache, so the immutable plan must also remain identical.
    bound_probe.pop("cache", None)
    payload["hardware_probe"] = bound_probe
    payload["plan_digest"] = "0" * 64
    payload["resolved_native_config"]["run"]["execution_plan_digest"] = None
    digest = _digest_payload(payload)
    payload["plan_digest"] = digest
    payload["resolved_native_config"]["run"]["execution_plan_digest"] = digest
    return ImmutableOPDPlan.model_validate(payload)
