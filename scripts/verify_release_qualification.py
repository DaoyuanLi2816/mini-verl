"""Fetch and validate a successful exact-SHA GPU qualification workflow artifact."""

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

from miniverl.qualification import sha256_file, validate_qualification_file

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
            raise ValueError(f"qualification artifact has too many files: {len(members)}")
        total = sum(member.file_size for member in members)
        if total > _MAX_UNCOMPRESSED_BYTES:
            raise ValueError(f"qualification artifact expands to too many bytes: {total}")
        for member in members:
            pure = PurePosixPath(member.filename.replace("\\", "/"))
            if pure.is_absolute() or ".." in pure.parts:
                raise ValueError(f"unsafe qualification archive member: {member.filename!r}")
            mode = member.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError(f"qualification archive contains a symlink: {member.filename!r}")
            target = destination.joinpath(*pure.parts)
            target.resolve().relative_to(destination.resolve())
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(member) as source, target.open("wb") as out:
                shutil.copyfileobj(source, out)


def fetch_and_validate(
    *,
    repository: str,
    workflow: str,
    commit: str,
    artifact_name: str,
    token: str,
    required_gpu_name: str,
    known_good: Path,
    output: Path,
) -> dict[str, Any]:
    api = f"https://api.github.com/repos/{repository}"
    query = (
        f"{api}/actions/workflows/{workflow}/runs?head_sha={commit}&status=completed&per_page=100"
    )
    runs = _request_json(query, token).get("workflow_runs") or []
    candidates = [
        run
        for run in runs
        if run.get("head_sha") == commit
        and run.get("conclusion") == "success"
        and run.get("event") == "workflow_dispatch"
    ]
    candidates.sort(key=lambda item: str(item.get("created_at", "")), reverse=True)
    if not candidates:
        raise RuntimeError(f"no successful {workflow} workflow_dispatch run exists for {commit}")
    known_good_sha = sha256_file(known_good)
    failures: list[str] = []
    for run in candidates:
        artifacts = (
            _request_json(f"{api}/actions/runs/{run['id']}/artifacts", token).get("artifacts") or []
        )
        matches = [
            item
            for item in artifacts
            if item.get("name") == artifact_name and not item.get("expired", False)
        ]
        for artifact in matches:
            with tempfile.TemporaryDirectory(prefix="miniverl-qualification-") as temporary:
                temporary_path = Path(temporary)
                archive = temporary_path / "artifact.zip"
                extracted = temporary_path / "extracted"
                extracted.mkdir()
                try:
                    _download(str(artifact["archive_download_url"]), token, archive)
                    _safe_extract(archive, extracted)
                    qualification = extracted / "qualification.json"
                    problems = validate_qualification_file(
                        qualification,
                        expected_commit=commit,
                        expected_known_good_sha256=known_good_sha,
                        required_gpu_name=required_gpu_name,
                    )
                    if problems:
                        failures.extend(f"run {run['id']}: {problem}" for problem in problems)
                        continue
                    if output.exists():
                        raise RuntimeError(f"output already exists: {output}")
                    shutil.copytree(extracted, output)
                    return {
                        "valid": True,
                        "source_commit": commit,
                        "workflow_run_id": run["id"],
                        "workflow_run_url": run["html_url"],
                        "artifact_id": artifact["id"],
                        "artifact_name": artifact_name,
                        "qualification_sha256": sha256_file(output / "qualification.json"),
                    }
                except (OSError, ValueError, zipfile.BadZipFile) as exc:
                    failures.append(f"run {run['id']}: {exc}")
    detail = "; ".join(failures) if failures else "matching artifact was not found"
    raise RuntimeError(f"no valid exact-SHA GPU qualification artifact: {detail}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--workflow", default="gpu.yml")
    parser.add_argument("--commit", required=True)
    parser.add_argument("--artifact-name", default="gpu-release-smoke")
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
            artifact_name=args.artifact_name,
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
