"""Fetch one same-run candidate and exact-SHA GPU qualification pair."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from miniverl.qualification import GPUQualification, sha256_file
from miniverl.release_candidate import load_candidate_manifest, validate_candidate_directory
from miniverl.release_chain import validate_release_chain

_MAX_FILES = 128
_MAX_UNCOMPRESSED_BYTES = 1024 * 1024 * 1024


def _request_json(url: str, token: str) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "miniVERL-release-qualification",
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def _download(url: str, token: str, destination: Path) -> None:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "miniVERL-release-qualification",
        },
    )
    with urllib.request.urlopen(request, timeout=180) as response, destination.open("wb") as out:
        shutil.copyfileobj(response, out)


def _safe_extract(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        if len(members) > _MAX_FILES:
            raise ValueError(f"artifact has too many files: {len(members)}")
        total = sum(member.file_size for member in members)
        if total > _MAX_UNCOMPRESSED_BYTES:
            raise ValueError(f"artifact expands to too many bytes: {total}")
        seen: set[str] = set()
        for member in members:
            pure = PurePosixPath(member.filename.replace("\\", "/"))
            normalized = pure.as_posix().rstrip("/")
            if (
                not normalized
                or pure.is_absolute()
                or ".." in pure.parts
                or any(part in {"", "."} for part in pure.parts)
            ):
                raise ValueError(f"unsafe artifact member: {member.filename!r}")
            if normalized in seen:
                raise ValueError(f"duplicate artifact member: {normalized!r}")
            seen.add(normalized)
            mode = member.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError(f"artifact contains a symlink: {member.filename!r}")
            target = destination.joinpath(*pure.parts)
            target.resolve().relative_to(destination.resolve())
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(member) as source, target.open("xb") as out:
                shutil.copyfileobj(source, out)


def _artifact(artifacts: list[dict[str, Any]], name: str) -> dict[str, Any]:
    matches = [item for item in artifacts if item.get("name") == name]
    if len(matches) != 1:
        raise ValueError(f"expected one unambiguous {name!r} artifact, found {len(matches)}")
    artifact = matches[0]
    if artifact.get("expired") is not False:
        raise ValueError(f"artifact {name!r} is expired")
    if not artifact.get("archive_download_url"):
        raise ValueError(f"artifact {name!r} has no download URL")
    return artifact


def _check_api_digest(artifact: dict[str, Any], archive: Path) -> None:
    digest = artifact.get("digest")
    if digest is None:
        return
    if not isinstance(digest, str) or not digest.startswith("sha256:"):
        raise ValueError("artifact API digest has an unsupported format")
    if sha256_file(archive) != digest.removeprefix("sha256:"):
        raise ValueError(f"artifact API digest mismatch for {artifact.get('name')!r}")


def _run_is_exact(
    run: dict[str, Any],
    *,
    repository: str,
    workflow_id: int,
    workflow_path: str,
    commit: str,
) -> bool:
    return (
        run.get("head_sha") == commit
        and run.get("conclusion") == "success"
        and run.get("event") == "workflow_dispatch"
        and run.get("workflow_id") == workflow_id
        and run.get("path") == workflow_path
        and (run.get("repository") or {}).get("full_name") == repository
        and (run.get("head_repository") or {}).get("full_name") == repository
    )


def fetch_and_validate(
    *,
    repository: str,
    workflow: str,
    commit: str,
    qualification_artifact_name: str,
    candidate_artifact_name: str,
    token: str,
    required_gpu_name: str,
    known_good: Path,
    output: Path,
) -> dict[str, Any]:
    api = f"https://api.github.com/repos/{repository}"
    workflow_data = _request_json(f"{api}/actions/workflows/{workflow}", token)
    workflow_id = int(workflow_data["id"])
    workflow_path = str(workflow_data["path"])
    query = (
        f"{api}/actions/workflows/{workflow}/runs?head_sha={commit}&status=completed&per_page=100"
    )
    runs = _request_json(query, token).get("workflow_runs") or []
    candidates = [
        run
        for run in runs
        if _run_is_exact(
            run,
            repository=repository,
            workflow_id=workflow_id,
            workflow_path=workflow_path,
            commit=commit,
        )
    ]
    candidates.sort(key=lambda item: str(item.get("created_at", "")), reverse=True)
    if not candidates:
        raise RuntimeError(f"no successful exact-repository {workflow} run exists for {commit}")
    known_good_sha = sha256_file(known_good)
    failures: list[str] = []
    for run in candidates:
        run_id = int(run["id"])
        try:
            artifacts = (
                _request_json(f"{api}/actions/runs/{run_id}/artifacts", token).get("artifacts")
                or []
            )
            candidate_artifact = _artifact(artifacts, candidate_artifact_name)
            qualification_artifact = _artifact(artifacts, qualification_artifact_name)
            with tempfile.TemporaryDirectory(prefix="miniverl-qualified-candidate-") as temporary:
                temporary_path = Path(temporary)
                candidate_zip = temporary_path / "candidate.zip"
                qualification_zip = temporary_path / "qualification.zip"
                candidate_root = temporary_path / "candidate"
                qualification_root = temporary_path / "qualification"
                candidate_root.mkdir()
                qualification_root.mkdir()
                _download(str(candidate_artifact["archive_download_url"]), token, candidate_zip)
                _download(
                    str(qualification_artifact["archive_download_url"]),
                    token,
                    qualification_zip,
                )
                _check_api_digest(candidate_artifact, candidate_zip)
                _check_api_digest(qualification_artifact, qualification_zip)
                _safe_extract(candidate_zip, candidate_root)
                _safe_extract(qualification_zip, qualification_root)
                candidate_problems = validate_candidate_directory(
                    candidate_root,
                    expected_commit=commit,
                    expected_repository=repository,
                    expected_workflow_path=workflow_path,
                    expected_run_id=run_id,
                    expected_run_attempt=int(run["run_attempt"]),
                )
                if candidate_problems:
                    raise ValueError("; ".join(candidate_problems))
                manifest_path = candidate_root / "candidate-manifest.json"
                manifest = load_candidate_manifest(manifest_path)
                manifest_sha = sha256_file(manifest_path)
                qualification_path = qualification_root / "qualification.json"
                qualification_problems = validate_release_chain(
                    candidate_root,
                    manifest_path,
                    qualification_path,
                    expected_commit=commit,
                    expected_known_good_sha256=known_good_sha,
                    required_gpu_name=required_gpu_name,
                )
                if qualification_problems:
                    raise ValueError("; ".join(qualification_problems))
                qualification = GPUQualification.model_validate(
                    json.loads(qualification_path.read_text(encoding="utf-8"))
                )
                if qualification.miniverl_version != manifest.miniverl_version:
                    raise ValueError("candidate and qualification versions differ")
                binding = qualification.candidate
                if (
                    binding.workflow_repository != repository
                    or binding.workflow_path != workflow_path
                    or binding.workflow_run_id != run_id
                    or binding.workflow_run_attempt != run.get("run_attempt")
                ):
                    raise ValueError("qualification is not bound to the selected workflow run")
                if output.exists():
                    raise RuntimeError(f"output already exists: {output}")
                output.mkdir(parents=True)
                shutil.copytree(candidate_root, output / "candidate")
                shutil.copytree(qualification_root, output / "qualification")
                result = {
                    "valid": True,
                    "repository": repository,
                    "workflow_id": workflow_id,
                    "workflow_path": workflow_path,
                    "workflow_run_id": run_id,
                    "workflow_run_attempt": run.get("run_attempt"),
                    "workflow_run_url": run["html_url"],
                    "source_commit": commit,
                    "candidate_artifact_id": candidate_artifact["id"],
                    "qualification_artifact_id": qualification_artifact["id"],
                    "candidate_manifest_sha256": manifest_sha,
                    "candidate_wheel_sha256": manifest.wheel.sha256,
                    "candidate_sdist_sha256": manifest.sdist.sha256,
                    "qualification_sha256": sha256_file(
                        output / "qualification/qualification.json"
                    ),
                }
                (output / "verification.json").write_text(
                    json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8",
                    newline="\n",
                )
                return result
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            failures.append(f"run {run_id}: {exc}")
    raise RuntimeError("no valid same-run candidate/qualification pair: " + "; ".join(failures))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--workflow", default="gpu.yml")
    parser.add_argument("--commit", required=True)
    parser.add_argument("--qualification-artifact-name", default="gpu-release-smoke")
    parser.add_argument("--candidate-artifact-name", default="candidate-distributions")
    parser.add_argument("--required-gpu-name", default="NVIDIA GeForce RTX 4080")
    parser.add_argument("--known-good", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--token-env", default="GITHUB_TOKEN")
    args = parser.parse_args()
    token = os.environ.get(args.token_env)
    if not token:
        parser.error(f"environment variable {args.token_env} is required")
    try:
        result = fetch_and_validate(
            repository=args.repository,
            workflow=args.workflow,
            commit=args.commit,
            qualification_artifact_name=args.qualification_artifact_name,
            candidate_artifact_name=args.candidate_artifact_name,
            token=token,
            required_gpu_name=args.required_gpu_name,
            known_good=args.known_good,
            output=args.output,
        )
    except (RuntimeError, urllib.error.URLError) as exc:
        print(json.dumps({"valid": False, "error": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
