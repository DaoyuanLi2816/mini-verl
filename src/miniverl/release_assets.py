"""Build and validate the canonical, torch-free GitHub Release asset set."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import re
import shutil
import stat
import tarfile
import tempfile
import unicodedata
from pathlib import Path, PurePosixPath
from typing import Any

from miniverl.qualification import GPUQualification, sha256_file, validate_qualification_payload
from miniverl.release_candidate import load_candidate_manifest, validate_candidate_directory
from miniverl.release_chain import validate_release_chain
from miniverl.utils.privacy import portable_text

__all__ = [
    "QUALIFICATION_SUM_FILES",
    "check_release_assets",
    "prepare_release_assets",
    "validate_canonical_names",
]

PRIMARY_EVIDENCE = {
    "release_smoke_record": "qualification-release-smoke.json",
    "full_direct_result": "qualification-direct-gkd.json",
    "full_pg_k1_result": "qualification-pg-k1.json",
    "full_smollm2_result": "qualification-smollm2.json",
}
ARCHIVE_EVIDENCE = {
    "adapter_adapter_config.json": ("adapter_config", "adapter/adapter_config.json"),
    "adapter_miniverl_adapter_manifest.json": (
        "adapter_manifest",
        "adapter/miniverl_adapter_manifest.json",
    ),
    "adapter_adapter_model.safetensors": (
        "adapter_weights",
        "adapter/adapter_model.safetensors",
    ),
    "inputs_prompts.parquet": ("input_prompts", "inputs/prompts.parquet"),
    "run_summary": ("run_summary", "run-summary.json"),
}
V011_ARCHIVE_EVIDENCE = {
    "full_v011_profiles_result": ("v011_profiles", "v011/profiles.json"),
    "full_hf_cached_runtime_result": (
        "v011_hf_cached_runtime",
        "v011/hf-cached-runtime.json",
    ),
    "full_vllm_runtime_result": ("v011_vllm_runtime", "v011/vllm-runtime.json"),
}
_SPECIAL_EVIDENCE = {"candidate-manifest.json"}
QUALIFICATION_SUM_FILES = (
    "candidate-manifest.json",
    "release-verification.json",
    "qualification.json",
    "qualification-release-smoke.json",
    "qualification-direct-gkd.json",
    "qualification-pg-k1.json",
    "qualification-smollm2.json",
    "qualification-evidence.tar.gz",
    "qualification-evidence-manifest.json",
)
_FIXED_FILES = {
    "SHA256SUMS",
    "candidate-manifest.json",
    "release-verification.json",
    "qualification.json",
    "qualification-SHA256SUMS",
    *QUALIFICATION_SUM_FILES[3:],
}
_DOUBLE_SUFFIX = re.compile(r"(?i)(\.json\.json|\.parquet\.parquet|\.safetensors\.safetensors)$")
_WINDOWS_RESERVED = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}


def _json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode()


def _private(payload: Any) -> bool:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return portable_text(serialized) != serialized


def validate_canonical_names(names: list[str] | tuple[str, ...]) -> None:
    """Reject names that are ambiguous on Windows or unsafe as release assets."""
    seen: set[str] = set()
    for name in names:
        if (
            not name
            or name != unicodedata.normalize("NFC", name)
            or not name.isascii()
            or "/" in name
            or "\\" in name
            or name.startswith(".")
            or name.endswith((".", " "))
            or ".." in name
            or _DOUBLE_SUFFIX.search(name)
        ):
            raise ValueError(f"noncanonical release filename: {name!r}")
        stem = name.split(".", 1)[0].upper()
        if stem in _WINDOWS_RESERVED:
            raise ValueError(f"reserved release filename: {name!r}")
        folded = unicodedata.normalize("NFC", name).casefold()
        if folded in seen:
            raise ValueError(f"case-insensitive release filename collision: {name!r}")
        seen.add(folded)


def _validate_archive_mapping() -> None:
    seen_paths: set[str] = set()
    seen_roles: set[str] = set()
    for original_name, (semantic_role, archive_path) in {
        **ARCHIVE_EVIDENCE,
        **V011_ARCHIVE_EVIDENCE,
    }.items():
        if (
            not original_name
            or not semantic_role.isascii()
            or not re.fullmatch(r"[a-z][a-z0-9_]*", semantic_role)
        ):
            raise ValueError(f"noncanonical archive semantic role: {semantic_role!r}")
        path = PurePosixPath(archive_path)
        if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
            raise ValueError(f"unsafe archive mapping path: {archive_path!r}")
        for part in path.parts:
            validate_canonical_names([part])
        folded_path = unicodedata.normalize("NFC", archive_path).casefold()
        if folded_path in seen_paths or semantic_role in seen_roles:
            raise ValueError("archive evidence mapping contains a collision")
        seen_paths.add(folded_path)
        seen_roles.add(semantic_role)


def _archive_evidence_for_version(version: str) -> dict[str, tuple[str, str]]:
    mapping = dict(ARCHIVE_EVIDENCE)
    version_parts = version.split(".", 2)
    try:
        is_v011 = (int(version_parts[0]), int(version_parts[1])) >= (0, 11)
    except (IndexError, ValueError):
        is_v011 = False
    if is_v011:
        mapping.update(V011_ARCHIVE_EVIDENCE)
    return mapping


def _safe_source(root: Path, relative: str) -> Path:
    candidate = root / relative
    try:
        metadata = candidate.lstat()
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise ValueError(f"unsafe or missing evidence path: {relative!r}") from exc
    attributes = getattr(metadata, "st_file_attributes", 0)
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    if stat.S_ISLNK(metadata.st_mode) or (reparse and attributes & reparse):
        raise ValueError(f"evidence must be a regular non-symlink file: {relative!r}")
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"evidence must be a regular file: {relative!r}")
    return resolved


def _verification(
    path: Path,
    *,
    manifest_sha: str,
    wheel_sha: str,
    sdist_sha: str,
    qualification_sha: str,
    source_commit: str,
    run_id: int,
    run_attempt: int,
) -> dict[str, Any]:
    try:
        payload = _json(path)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid release verification: {exc}") from exc
    if not isinstance(payload, dict) or _private(payload):
        raise ValueError("release verification failed privacy or object validation")
    expected = {
        "candidate_manifest_sha256": manifest_sha,
        "candidate_wheel_sha256": wheel_sha,
        "candidate_sdist_sha256": sdist_sha,
        "qualification_sha256": qualification_sha,
        "source_commit": source_commit,
        "workflow_run_id": run_id,
        "workflow_run_attempt": run_attempt,
        "valid": True,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"release verification {key} is not bound to the accepted chain")
    return payload


def _write_archive(destination: Path, members: list[tuple[str, str, str, Path, str, int]]) -> None:
    raw = io.BytesIO()
    with (
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9) as zipped,
        tarfile.open(fileobj=zipped, mode="w", format=tarfile.PAX_FORMAT) as archive,
    ):
        for _, archive_path, _, source, _, size in sorted(members, key=lambda item: item[1]):
            info = tarfile.TarInfo(archive_path)
            info.size = size
            info.mode = 0o644
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            with source.open("rb") as handle:
                archive.addfile(info, handle)
    destination.write_bytes(raw.getvalue())


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def prepare_release_assets(
    candidate_dir: str | Path,
    candidate_manifest: str | Path,
    qualification_root: str | Path,
    qualification: str | Path,
    verification: str | Path,
    output: str | Path,
) -> None:
    """Validate one accepted full chain and atomically publish its canonical assets."""
    candidate_root = Path(candidate_dir).resolve(strict=True)
    manifest_path = Path(candidate_manifest).resolve(strict=True)
    evidence_root = Path(qualification_root).resolve(strict=True)
    qualification_path = Path(qualification).resolve(strict=True)
    verification_path = Path(verification).resolve(strict=True)
    target = Path(output)
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise ValueError("release asset output must initially be empty")
    if qualification_path != evidence_root / "qualification.json":
        raise ValueError("qualification must be qualification-root/qualification.json")
    _validate_archive_mapping()
    candidate_problems = validate_candidate_directory(candidate_root, manifest_path=manifest_path)
    if candidate_problems:
        raise ValueError("candidate validation failed: " + "; ".join(candidate_problems))
    candidate = load_candidate_manifest(manifest_path)
    try:
        qualification_payload = _json(qualification_path)
        record = GPUQualification.model_validate(qualification_payload)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"qualification validation failed: {exc}") from exc
    if record.level != "full_qualification":
        raise ValueError("qualification level must be full_qualification")
    chain_problems = validate_release_chain(
        candidate_root,
        manifest_path,
        qualification_path,
        expected_commit=record.source_commit,
        expected_known_good_sha256=record.environment.known_good_manifest_sha256,
        required_gpu_name=record.environment.gpu_name,
    )
    if chain_problems:
        raise ValueError("release chain validation failed: " + "; ".join(chain_problems))
    artifact_problems = validate_qualification_payload(
        qualification_payload,
        artifact_root=evidence_root,
    )
    if artifact_problems:
        raise ValueError(
            "qualification evidence validation failed: " + "; ".join(artifact_problems)
        )

    artifacts = {item.name: item for item in record.artifacts}
    archive_evidence = _archive_evidence_for_version(record.miniverl_version)
    expected_roles = set(PRIMARY_EVIDENCE) | set(archive_evidence) | _SPECIAL_EVIDENCE
    unknown = sorted(set(artifacts) - expected_roles)
    missing = sorted(expected_roles - set(artifacts))
    if unknown:
        raise ValueError("unknown evidence role: " + ", ".join(unknown))
    if missing:
        raise ValueError("required evidence role is missing: " + ", ".join(missing))
    if len(artifacts) != len(record.artifacts):
        raise ValueError("duplicate evidence roles are not allowed")
    manifest_evidence = artifacts["candidate-manifest.json"]
    manifest_evidence_path = _safe_source(evidence_root, manifest_evidence.path)
    if manifest_evidence_path.read_bytes() != manifest_path.read_bytes():
        raise ValueError("qualification candidate-manifest evidence differs from the candidate")
    if candidate.workflow.run_id is None or candidate.workflow.run_attempt is None:
        raise ValueError("release assets require a GitHub Actions run id and attempt")

    qualification_sha = sha256_file(qualification_path)
    verification_payload = _verification(
        verification_path,
        manifest_sha=sha256_file(manifest_path),
        wheel_sha=candidate.wheel.sha256,
        sdist_sha=candidate.sdist.sha256,
        qualification_sha=qualification_sha,
        source_commit=candidate.source_commit,
        run_id=candidate.workflow.run_id,
        run_attempt=candidate.workflow.run_attempt,
    )
    validate_canonical_names(list(_FIXED_FILES))

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".release-assets-", dir=target.parent))
    try:
        for distribution in (candidate.wheel, candidate.sdist):
            _copy(
                _safe_source(candidate_root, distribution.filename),
                staging / "dist" / distribution.filename,
            )
        _copy(candidate_root / "SHA256SUMS", staging / "SHA256SUMS")
        _copy(manifest_path, staging / "candidate-manifest.json")
        (staging / "release-verification.json").write_bytes(_json_bytes(verification_payload))
        _copy(qualification_path, staging / "qualification.json")
        for role, destination in PRIMARY_EVIDENCE.items():
            artifact = artifacts[role]
            _copy(_safe_source(evidence_root, artifact.path), staging / destination)

        archive_members: list[tuple[str, str, str, Path, str, int]] = []
        for original_name, (semantic_role, member_path) in archive_evidence.items():
            artifact = artifacts[original_name]
            source = _safe_source(evidence_root, artifact.path)
            archive_members.append(
                (
                    semantic_role,
                    member_path,
                    original_name,
                    source,
                    artifact.sha256,
                    artifact.bytes,
                )
            )
        archive_output = staging / "qualification-evidence.tar.gz"
        _write_archive(archive_output, archive_members)
        archive_manifest = {
            "schema_version": 1,
            "kind": "miniverl_qualification_evidence_manifest",
            "miniverl_version": candidate.miniverl_version,
            "source_commit": candidate.source_commit,
            "qualification_sha256": qualification_sha,
            "archive": {
                "filename": archive_output.name,
                "sha256": sha256_file(archive_output),
                "bytes": archive_output.stat().st_size,
            },
            "members": [
                {
                    "semantic_role": role,
                    "archive_path": member_path,
                    "original_evidence_name": original_name,
                    "sha256": digest,
                    "bytes": size,
                }
                for role, member_path, original_name, _, digest, size in sorted(
                    archive_members, key=lambda item: item[1]
                )
            ],
        }
        (staging / "qualification-evidence-manifest.json").write_bytes(
            _json_bytes(archive_manifest)
        )
        (staging / "qualification-SHA256SUMS").write_text(
            "".join(f"{sha256_file(staging / name)}  {name}\n" for name in QUALIFICATION_SUM_FILES),
            encoding="utf-8",
            newline="\n",
        )
        problems = check_release_assets(staging)
        if problems:
            raise ValueError("built release assets failed validation: " + "; ".join(problems))
        if target.exists():
            target.rmdir()
        staging.replace(target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _checksum_problems(
    root: Path, sums_name: str, expected_names: tuple[str, ...] | None
) -> list[str]:
    problems: list[str] = []
    try:
        lines = (root / sums_name).read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        return [f"cannot read {sums_name}: {exc}"]
    names: list[str] = []
    for line in lines:
        match = re.fullmatch(r"([0-9a-f]{64})  ([^/\\]+)", line)
        if not match:
            problems.append(f"invalid {sums_name} line: {line!r}")
            continue
        digest, name = match.groups()
        names.append(name)
        path = root / name
        if not path.is_file() or path.is_symlink():
            problems.append(f"{sums_name} references missing or unsafe file {name}")
        elif sha256_file(path) != digest:
            problems.append(f"{sums_name} checksum mismatch for {name}")
    if expected_names is not None and names != list(expected_names):
        problems.append(f"{sums_name} does not contain the canonical ordered file set")
    return problems


def check_release_assets(output: str | Path) -> list[str]:
    """Purely validate an already-built canonical asset directory."""
    root = Path(output)
    if not root.is_dir() or root.is_symlink():
        return ["release asset output is not a regular directory"]
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    actual_directories = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_dir()
    }
    dist = root / "dist"
    dist_names = sorted(path.name for path in dist.iterdir()) if dist.is_dir() else []
    expected_files = _FIXED_FILES | {f"dist/{name}" for name in dist_names}
    problems: list[str] = []
    try:
        _validate_archive_mapping()
    except ValueError as exc:
        problems.append(str(exc))
    if actual_directories != {"dist"}:
        problems.append(
            "unexpected release asset directory set: "
            + ", ".join(sorted(actual_directories ^ {"dist"}))
        )
    unsafe_files = sorted(
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_symlink()
    )
    if unsafe_files:
        problems.append("release assets must not contain symlinks: " + ", ".join(unsafe_files))
    if actual_files != expected_files:
        problems.append(
            "unexpected release asset file set: " + ", ".join(sorted(actual_files ^ expected_files))
        )
    try:
        validate_canonical_names([*(_FIXED_FILES), *dist_names])
    except ValueError as exc:
        problems.append(str(exc))
    if (
        len(dist_names) != 2
        or sum(name.endswith(".whl") for name in dist_names) != 1
        or sum(name.endswith(".tar.gz") for name in dist_names) != 1
    ):
        problems.append("dist must contain exactly one wheel and one sdist")
    problems.extend(_checksum_problems(root, "qualification-SHA256SUMS", QUALIFICATION_SUM_FILES))
    # Distribution sums live one directory above dist and intentionally name only distributions.
    if dist.is_dir():
        try:
            sums = (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
            declared = []
            for line in sums:
                digest, name = line.split("  ", 1)
                declared.append(name)
                if not re.fullmatch(r"[0-9a-f]{64}", digest) or sha256_file(dist / name) != digest:
                    problems.append(f"SHA256SUMS checksum mismatch for {name}")
            if sorted(declared) != dist_names:
                problems.append("SHA256SUMS does not cover the exact dist file set")
        except (OSError, ValueError, IndexError) as exc:
            problems.append(f"invalid SHA256SUMS: {exc}")

    qualification_record: GPUQualification | None = None
    candidate_record = None
    qualification_artifacts: dict[str, Any] = {}
    try:
        candidate_record = load_candidate_manifest(root / "candidate-manifest.json")
        qualification_payload = _json(root / "qualification.json")
        qualification_record = GPUQualification.model_validate(qualification_payload)
        if qualification_record.level != "full_qualification":
            problems.append("qualification level is not full_qualification")
        qualification_artifacts = {
            artifact.name: artifact for artifact in qualification_record.artifacts
        }
        archive_evidence = _archive_evidence_for_version(qualification_record.miniverl_version)
        expected_roles = set(PRIMARY_EVIDENCE) | set(archive_evidence) | _SPECIAL_EVIDENCE
        if set(qualification_artifacts) != expected_roles or len(qualification_artifacts) != len(
            qualification_record.artifacts
        ):
            problems.append("qualification evidence roles are not the canonical exact set")
        problems.extend(
            validate_qualification_payload(
                qualification_payload,
                expected_commit=candidate_record.source_commit,
                expected_wheel_sha256=candidate_record.wheel.sha256,
                expected_candidate_manifest_sha256=sha256_file(root / "candidate-manifest.json"),
                expected_known_good_sha256=(
                    qualification_record.environment.known_good_manifest_sha256
                ),
                required_gpu_name=qualification_record.environment.gpu_name,
            )
        )
        bindings = (
            (
                candidate_record.miniverl_version,
                qualification_record.miniverl_version,
                "version",
            ),
            (
                candidate_record.workflow.run_id,
                qualification_record.candidate.workflow_run_id,
                "run id",
            ),
            (
                candidate_record.workflow.run_attempt,
                qualification_record.candidate.workflow_run_attempt,
                "run attempt",
            ),
        )
        for candidate_value, qualification_value, label in bindings:
            if candidate_value != qualification_value:
                problems.append(f"candidate and qualification {label} binding differs")
        expected_dist = {candidate_record.wheel.filename, candidate_record.sdist.filename}
        if set(dist_names) != expected_dist:
            problems.append("dist filenames do not match the candidate manifest")
        for distribution in (candidate_record.wheel, candidate_record.sdist):
            distribution_path = dist / distribution.filename
            if distribution_path.is_file() and (
                distribution_path.stat().st_size != distribution.bytes
                or sha256_file(distribution_path) != distribution.sha256
            ):
                problems.append(
                    f"candidate distribution byte identity mismatch: {distribution.filename}"
                )
        candidate_evidence = qualification_artifacts.get("candidate-manifest.json")
        if candidate_evidence is None or (
            candidate_evidence.bytes != (root / "candidate-manifest.json").stat().st_size
            or candidate_evidence.sha256 != sha256_file(root / "candidate-manifest.json")
        ):
            problems.append("qualification evidence does not bind candidate-manifest.json")
        for role, destination in PRIMARY_EVIDENCE.items():
            artifact = qualification_artifacts.get(role)
            destination_path = root / destination
            if artifact is None or (
                destination_path.stat().st_size != artifact.bytes
                or sha256_file(destination_path) != artifact.sha256
            ):
                problems.append(f"qualification evidence byte identity mismatch: {role}")
        if (
            candidate_record.workflow.run_id is None
            or candidate_record.workflow.run_attempt is None
        ):
            problems.append("candidate has no GitHub Actions run identity")
        else:
            _verification(
                root / "release-verification.json",
                manifest_sha=sha256_file(root / "candidate-manifest.json"),
                wheel_sha=candidate_record.wheel.sha256,
                sdist_sha=candidate_record.sdist.sha256,
                qualification_sha=sha256_file(root / "qualification.json"),
                source_commit=candidate_record.source_commit,
                run_id=candidate_record.workflow.run_id,
                run_attempt=candidate_record.workflow.run_attempt,
            )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, KeyError) as exc:
        problems.append(f"invalid candidate, qualification or verification binding: {exc}")

    archive_path = root / "qualification-evidence.tar.gz"
    try:
        manifest = _json(root / "qualification-evidence-manifest.json")
        if _private(manifest):
            problems.append("archive manifest contains private data")
        declared_archive = manifest["archive"]
        if declared_archive["filename"] != archive_path.name:
            problems.append("archive manifest filename mismatch")
        if declared_archive["sha256"] != sha256_file(archive_path):
            problems.append("archive manifest checksum mismatch")
        if declared_archive["bytes"] != archive_path.stat().st_size:
            problems.append("archive manifest size mismatch")
        if qualification_record is not None and candidate_record is not None:
            if (
                manifest.get("schema_version") != 1
                or manifest.get("kind") != "miniverl_qualification_evidence_manifest"
                or manifest.get("miniverl_version") != candidate_record.miniverl_version
                or manifest.get("source_commit") != candidate_record.source_commit
                or manifest.get("qualification_sha256") != sha256_file(root / "qualification.json")
            ):
                problems.append("archive manifest release-chain binding mismatch")
            expected_member_rows = []
            archive_evidence = _archive_evidence_for_version(qualification_record.miniverl_version)
            for original_name, (semantic_role, member_path) in archive_evidence.items():
                artifact = qualification_artifacts.get(original_name)
                if artifact is not None:
                    expected_member_rows.append(
                        {
                            "semantic_role": semantic_role,
                            "archive_path": member_path,
                            "original_evidence_name": original_name,
                            "sha256": artifact.sha256,
                            "bytes": artifact.bytes,
                        }
                    )
            expected_member_rows.sort(key=lambda item: item["archive_path"])
            if manifest.get("members") != expected_member_rows:
                problems.append("archive manifest semantic mapping does not match qualification")
        declared_members = {item["archive_path"]: item for item in manifest["members"]}
        if len(declared_members) != len(manifest["members"]):
            problems.append("duplicate archive manifest member")
        names: set[str] = set()
        with tarfile.open(archive_path, "r:gz") as archive:
            for member in archive.getmembers():
                tar_member_path = PurePosixPath(member.name)
                if tar_member_path.is_absolute() or ".." in tar_member_path.parts:
                    problems.append(f"unsafe archive member: {member.name}")
                if member.name in names:
                    problems.append(f"duplicate archive member: {member.name}")
                names.add(member.name)
                if not member.isfile():
                    problems.append(f"non-regular archive member: {member.name}")
                if (
                    any((member.uid, member.gid, member.mtime))
                    or member.uname
                    or member.gname
                    or member.mode != 0o644
                ):
                    problems.append(f"nondeterministic archive metadata: {member.name}")
                extracted = archive.extractfile(member)
                data = extracted.read() if extracted is not None else b""
                member_declaration = declared_members.get(member.name)
                if member_declaration is None:
                    problems.append(f"undeclared archive member: {member.name}")
                elif (
                    member_declaration["bytes"] != len(data)
                    or member_declaration["sha256"] != hashlib.sha256(data).hexdigest()
                ):
                    problems.append(f"archive member byte identity mismatch: {member.name}")
        if names != set(declared_members):
            problems.append("archive member set does not match its manifest")
    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        tarfile.TarError,
        json.JSONDecodeError,
    ) as exc:
        problems.append(f"invalid evidence archive or manifest: {exc}")
    return problems
