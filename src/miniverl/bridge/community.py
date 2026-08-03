"""Privacy-safe community hardware/recipe submission records."""

from __future__ import annotations

import hashlib
import json
import platform
import re
from importlib import resources
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from miniverl import __version__
from miniverl.bridge.contract import BRIDGE_PROFILE, VERL_TAG
from miniverl.utils.privacy import portable_text
from miniverl.utils.runs import read_json, utc_now, write_json

__all__ = ["export_community_submission", "load_recipe_registry", "validate_submission"]

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class _Artifact(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    public_url: str | None = None


class _Submission(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: int = 1
    created_at: str
    miniverl_version: str
    recipe_id: str
    recipe_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    measured_status: str = Field(pattern=r"^(measured|not_measured)$")
    hardware: dict[str, Any]
    software: dict[str, Any]
    wall_time_seconds: float | None = Field(default=None, ge=0)
    benchmark: str
    artifacts: list[_Artifact]
    compatible_miniverl_release: str
    compatible_verl_release: str
    compatible_verl_bridge_profile: str
    notes: str = ""


class _RecipeArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class _RecipeRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: int = 1
    recipe_id: str
    category: str
    method: str
    model: dict[str, Any]
    hardware: dict[str, Any]
    wall_time_seconds: float | None = Field(default=None, ge=0)
    benchmark: str
    artifact: _RecipeArtifact
    measured_status: str = Field(pattern=r"^(measured|not_measured)$")
    compatible_miniverl_release: str
    compatible_verl_bridge_profile: str
    notes: str = ""


def _recipe_path(recipe_id: str) -> Path:
    package_root = resources.files("miniverl.community")
    candidate = package_root.joinpath("recipes", "v1", f"{recipe_id}.yaml")
    path = Path(str(candidate))
    if not path.is_file():
        raise ValueError(f"unknown community recipe {recipe_id!r}")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_recipe_registry() -> list[dict[str, Any]]:
    """Load and schema-validate every packaged version-1 recipe record."""
    import yaml

    package_root = resources.files("miniverl.community").joinpath("recipes", "v1")
    records: list[dict[str, Any]] = []
    for resource in sorted(package_root.iterdir(), key=lambda item: item.name):
        if not resource.name.endswith(".yaml"):
            continue
        payload = yaml.safe_load(resource.read_text(encoding="utf-8"))
        record = _RecipeRecord.model_validate(payload)
        if record.measured_status == "measured" and record.wall_time_seconds is None:
            raise ValueError(f"measured recipe {record.recipe_id!r} has no wall time")
        records.append(record.model_dump(mode="json"))
    return records


def export_community_submission(
    out: str | Path,
    *,
    recipe_id: str = "recoverybench",
) -> dict[str, Any]:
    """Write an honest unmeasured template bound to a packaged recipe."""
    recipe = _recipe_path(recipe_id)
    payload = _Submission(
        created_at=utc_now(),
        miniverl_version=__version__,
        recipe_id=recipe_id,
        recipe_sha256=_sha256(recipe),
        measured_status="not_measured",
        hardware={
            "gpu_name": "not reported",
            "gpu_vram_gib": None,
            "gpu_count": 1,
        },
        software={
            "python": platform.python_version(),
            "os": platform.system(),
            "miniverl": __version__,
        },
        wall_time_seconds=None,
        benchmark="not run; submission template only",
        artifacts=[],
        compatible_miniverl_release="0.6.0",
        compatible_verl_release=VERL_TAG,
        compatible_verl_bridge_profile=BRIDGE_PROFILE,
        notes="Fill measured fields only from retained artifacts; do not add local paths or secrets.",
    ).model_dump(mode="json")
    write_json(out, payload)
    return payload


def validate_submission(path: str | Path) -> list[str]:
    """Validate schema, recipe identity, artifacts and privacy."""
    problems: list[str] = []
    try:
        payload = read_json(path)
        submission = _Submission.model_validate(payload)
    except (OSError, json.JSONDecodeError, ValidationError, ValueError) as exc:
        return [f"schema: {exc}"]
    try:
        recipe = _recipe_path(submission.recipe_id)
    except ValueError as exc:
        problems.append(str(exc))
    else:
        if submission.recipe_sha256 != _sha256(recipe):
            problems.append("recipe_sha256 does not match the packaged recipe")
    if submission.measured_status == "measured":
        if submission.wall_time_seconds is None:
            problems.append("measured submissions require wall_time_seconds")
        if not submission.artifacts:
            problems.append("measured submissions require artifact hashes")
    for artifact in submission.artifacts:
        if not _SHA256.fullmatch(artifact.sha256):
            problems.append(f"artifact {artifact.name} has an invalid SHA-256")
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    if portable_text(serialized) != serialized:
        problems.append(
            "privacy: submission contains a local path, credential, or environment reference"
        )
    return problems
