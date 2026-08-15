"""Torch-free release-candidate construction and byte-identity validation."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from miniverl import __version__
from miniverl.utils.privacy import portable_text

__all__ = [
    "CandidateManifest",
    "PINNED_BUILD_TOOLS",
    "build_release_candidate",
    "candidate_manifest_sha256",
    "load_candidate_manifest",
    "sha256_file",
    "validate_candidate_directory",
]

PINNED_BUILD_TOOLS = {"build": "1.5.0", "hatchling": "1.32.0", "twine": "7.0.0"}
_SHA1 = r"^[0-9a-f]{40}$"
_SHA256 = r"^[0-9a-f]{64}$"
_PRIVATE = re.compile(
    r"(?i)(?:[a-z]:[\\/](?:users|documents)[\\/]|/home/|/users/|onedrive|"
    r"(?:token|credential|password|secret)\s*[:=])"
)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class DistributionRecord(_Strict):
    filename: str = Field(min_length=1, pattern=r"^[^/\\]+$")
    bytes: int = Field(gt=0)
    sha256: str = Field(pattern=_SHA256)


class BuildEnvironment(_Strict):
    os: str = Field(min_length=1)
    python: str = Field(min_length=1)
    tools: dict[str, str]

    @model_validator(mode="after")
    def _tools_are_exact(self) -> BuildEnvironment:
        if self.tools != PINNED_BUILD_TOOLS:
            raise ValueError("build tools do not match the pinned candidate toolchain")
        return self


class WorkflowContext(_Strict):
    kind: Literal["local", "github_actions"]
    repository: str | None = None
    workflow_path: str | None = None
    run_id: int | None = Field(default=None, gt=0)
    run_attempt: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _github_context_is_complete(self) -> WorkflowContext:
        values = (self.repository, self.workflow_path, self.run_id, self.run_attempt)
        if self.kind == "github_actions" and any(value is None for value in values):
            raise ValueError("GitHub Actions candidate context is incomplete")
        if self.kind == "local" and any(value is not None for value in values):
            raise ValueError("local candidate context must not claim workflow identity")
        return self


class CandidateManifest(_Strict):
    schema_version: Literal[1] = 1
    kind: Literal["miniverl_release_candidate"] = "miniverl_release_candidate"
    source_commit: str = Field(pattern=_SHA1)
    miniverl_version: str = Field(min_length=1)
    created_at: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
    artifact_name: Literal["candidate-distributions"] = "candidate-distributions"
    workflow: WorkflowContext
    build: BuildEnvironment
    wheel: DistributionRecord
    sdist: DistributionRecord

    @model_validator(mode="after")
    def _distribution_names_are_unambiguous(self) -> CandidateManifest:
        if not self.wheel.filename.endswith(".whl"):
            raise ValueError("candidate wheel filename must end in .whl")
        if not self.sdist.filename.endswith(".tar.gz"):
            raise ValueError("candidate sdist filename must end in .tar.gz")
        if self.wheel.filename == self.sdist.filename:
            raise ValueError("candidate distribution filenames must differ")
        return self


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_manifest_sha256(path: str | Path) -> str:
    return sha256_file(path)


def _finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    return True


def _json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def load_candidate_manifest(path: str | Path) -> CandidateManifest:
    payload = _json(Path(path))
    if not _finite(payload):
        raise ValueError("candidate manifest contains NaN or infinity")
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False)
    if portable_text(text) != text or _PRIVATE.search(text):
        raise ValueError("candidate manifest contains private or non-portable text")
    return CandidateManifest.model_validate(payload)


def _expected_sums(record: CandidateManifest) -> str:
    return "".join(f"{item.sha256}  {item.filename}\n" for item in (record.wheel, record.sdist))


def validate_candidate_directory(
    directory: str | Path,
    *,
    manifest_path: str | Path | None = None,
    expected_commit: str | None = None,
    expected_version: str | None = None,
    expected_repository: str | None = None,
    expected_workflow_path: str | None = None,
    expected_run_id: int | None = None,
    expected_run_attempt: int | None = None,
) -> list[str]:
    """Validate one closed candidate directory; return an empty list on success."""
    root = Path(directory)
    try:
        resolved_root = root.resolve(strict=True)
    except OSError as exc:
        return [f"candidate: {exc}"]
    manifest = Path(manifest_path) if manifest_path else root / "candidate-manifest.json"
    try:
        resolved_manifest = manifest.resolve(strict=True)
        resolved_manifest.relative_to(resolved_root)
        record = load_candidate_manifest(resolved_manifest)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        return [f"manifest: {exc}"]
    expected_names = {
        "candidate-manifest.json",
        "SHA256SUMS",
        record.wheel.filename,
        record.sdist.filename,
    }
    problems: list[str] = []
    try:
        entries = list(root.iterdir())
    except OSError as exc:
        return [f"candidate: {exc}"]
    actual_names = {entry.name for entry in entries}
    if actual_names != expected_names:
        problems.append(
            f"candidate: expected exactly {sorted(expected_names)}, got {sorted(actual_names)}"
        )
    for entry in entries:
        if entry.is_symlink() or not entry.is_file():
            problems.append(f"candidate: {entry.name!r} must be a regular non-symlink file")
    if resolved_manifest.name != "candidate-manifest.json":
        problems.append("manifest: filename must be candidate-manifest.json")
    bindings = (
        (expected_commit, record.source_commit, "source commit"),
        (expected_version, record.miniverl_version, "version"),
        (expected_repository, record.workflow.repository, "repository"),
        (expected_workflow_path, record.workflow.workflow_path, "workflow path"),
        (expected_run_id, record.workflow.run_id, "workflow run id"),
        (expected_run_attempt, record.workflow.run_attempt, "workflow run attempt"),
    )
    for expected, actual, label in bindings:
        if expected is not None and actual != expected:
            problems.append(f"binding: candidate {label} {actual!r} does not match {expected!r}")
    for item in (record.wheel, record.sdist):
        path = root / item.filename
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(resolved_root)
            if resolved.is_symlink() or not resolved.is_file():
                raise ValueError("not a regular file")
            if resolved.stat().st_size != item.bytes:
                problems.append(f"candidate: size mismatch for {item.filename}")
            actual = sha256_file(resolved)
            if actual != item.sha256:
                problems.append(f"candidate: checksum mismatch for {item.filename}")
        except (OSError, ValueError) as exc:
            problems.append(f"candidate: unsafe or missing {item.filename!r}: {exc}")
    sums = root / "SHA256SUMS"
    try:
        if sums.is_symlink() or sums.read_text(encoding="utf-8") != _expected_sums(record):
            problems.append("candidate: SHA256SUMS is not canonical")
    except (OSError, UnicodeError) as exc:
        problems.append(f"candidate: SHA256SUMS cannot be read: {exc}")
    return problems


def _workflow_context(env: dict[str, str]) -> WorkflowContext:
    if not env.get("GITHUB_ACTIONS"):
        return WorkflowContext(kind="local")
    workflow_ref = env.get("GITHUB_WORKFLOW_REF", "")
    workflow_path = workflow_ref.split("@", 1)[0].split("/", 2)[-1]
    return WorkflowContext(
        kind="github_actions",
        repository=env.get("GITHUB_REPOSITORY"),
        workflow_path=workflow_path,
        run_id=int(env["GITHUB_RUN_ID"]),
        run_attempt=int(env["GITHUB_RUN_ATTEMPT"]),
    )


def _tool_versions() -> dict[str, str]:
    versions = {name: importlib.metadata.version(name) for name in PINNED_BUILD_TOOLS}
    if versions != PINNED_BUILD_TOOLS:
        raise RuntimeError(f"candidate build tool mismatch: {versions!r}")
    return versions


def build_release_candidate(
    output: str | Path,
    *,
    source_commit: str,
    project_root: str | Path,
    env: dict[str, str] | None = None,
) -> CandidateManifest:
    """Build wheel+sdist once and close the output as a validated candidate."""
    root = Path(project_root).resolve(strict=True)
    destination = Path(output)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("candidate output directory must be empty")
    destination.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise ValueError("candidate output directory must not be a symlink")
    if not re.fullmatch(_SHA1[1:-1], source_commit):
        raise ValueError("source commit must be a full lowercase Git SHA")
    tools = _tool_versions()
    subprocess.run(
        [sys.executable, "-m", "build", "--no-isolation", "--outdir", str(destination)],
        cwd=root,
        check=True,
    )
    files = [entry for entry in destination.iterdir() if entry.is_file()]
    wheels = [entry for entry in files if entry.suffix == ".whl"]
    sdists = [entry for entry in files if entry.name.endswith(".tar.gz")]
    if len(files) != 2 or len(wheels) != 1 or len(sdists) != 1:
        raise RuntimeError("candidate build must produce exactly one wheel and one sdist")
    subprocess.run(
        [sys.executable, "-m", "twine", "check", str(wheels[0]), str(sdists[0])],
        cwd=root,
        check=True,
    )
    record = CandidateManifest(
        source_commit=source_commit,
        miniverl_version=__version__,
        created_at=datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        workflow=_workflow_context(dict(os.environ) if env is None else env),
        build=BuildEnvironment(os=platform.system(), python=platform.python_version(), tools=tools),
        wheel=DistributionRecord(
            filename=wheels[0].name,
            bytes=wheels[0].stat().st_size,
            sha256=sha256_file(wheels[0]),
        ),
        sdist=DistributionRecord(
            filename=sdists[0].name,
            bytes=sdists[0].stat().st_size,
            sha256=sha256_file(sdists[0]),
        ),
    )
    manifest = destination / "candidate-manifest.json"
    manifest.write_text(
        json.dumps(record.model_dump(mode="json"), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (destination / "SHA256SUMS").write_text(_expected_sums(record), encoding="utf-8", newline="\n")
    problems = validate_candidate_directory(
        destination, expected_commit=source_commit, expected_version=__version__
    )
    if problems:
        raise RuntimeError("built candidate failed validation: " + "; ".join(problems))
    return record
