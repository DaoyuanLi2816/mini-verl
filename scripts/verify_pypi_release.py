"""Verify that one PyPI release matches locally built distributions exactly."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

_INTEGRITY_MEDIA_TYPE = "application/vnd.pypi.integrity.v1+json"


def _request_json(url: str, *, accept: str = "application/json") -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={"Accept": accept, "User-Agent": "miniVERL-release-verifier/0.2"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def _expected_hashes(sums: Path, artifact_root: Path) -> dict[str, str]:
    expected: dict[str, str] = {}
    for line in sums.read_text(encoding="ascii").splitlines():
        digest, relative = line.split(maxsplit=1)
        path = artifact_root / relative.strip().lstrip("*")
        if not path.is_file():
            raise RuntimeError(f"SHA256SUMS references missing artifact: {path}")
        expected[path.name] = digest
    wheels = [name for name in expected if name.endswith(".whl")]
    sdists = [name for name in expected if name.endswith(".tar.gz")]
    if len(expected) != 2 or len(wheels) != 1 or len(sdists) != 1:
        raise RuntimeError(
            f"expected exactly one wheel and one sdist in SHA256SUMS, found {sorted(expected)}"
        )
    return expected


def _verify_metadata(
    *,
    project: str,
    version: str,
    expected: dict[str, str],
) -> list[str]:
    metadata = _request_json(f"https://pypi.org/pypi/{project}/{version}/json")
    info = metadata.get("info") or {}
    if info.get("name") != project:
        raise RuntimeError(f"PyPI project name is {info.get('name')!r}, expected {project!r}")
    if info.get("version") != version:
        raise RuntimeError(f"PyPI version is {info.get('version')!r}, expected {version!r}")

    files = {str(item.get("filename")): item for item in metadata.get("urls") or []}
    if set(files) != set(expected):
        raise RuntimeError(
            f"PyPI files {sorted(files)} do not match built files {sorted(expected)}"
        )
    file_types = {str(item.get("packagetype")) for item in files.values()}
    if file_types != {"bdist_wheel", "sdist"}:
        raise RuntimeError(f"unexpected PyPI file types: {sorted(file_types)}")

    urls: list[str] = []
    for filename, digest in expected.items():
        item = files[filename]
        actual = str((item.get("digests") or {}).get("sha256") or "")
        if actual != digest:
            raise RuntimeError(
                f"PyPI SHA-256 mismatch for {filename}: expected {digest}, got {actual}"
            )
        url = str(item.get("url") or "")
        if not url.startswith("https://files.pythonhosted.org/"):
            raise RuntimeError(f"unexpected download URL for {filename}: {url!r}")
        urls.append(url)
    return urls


def _verify_integrity_metadata(*, project: str, version: str, filenames: list[str]) -> None:
    for filename in filenames:
        encoded = urllib.parse.quote(filename, safe="")
        provenance = _request_json(
            f"https://pypi.org/integrity/{project}/{version}/{encoded}/provenance",
            accept=_INTEGRITY_MEDIA_TYPE,
        )
        bundles = provenance.get("attestation_bundles")
        if not isinstance(bundles, list) or not bundles:
            raise RuntimeError(f"PyPI exposes no attestation bundle for {filename}")
        if not any(bundle.get("attestations") for bundle in bundles if isinstance(bundle, dict)):
            raise RuntimeError(f"PyPI exposes no attestation for {filename}")


def verify_release(
    *,
    project: str,
    version: str,
    sums: Path,
    artifact_root: Path,
    repository: str,
    attempts: int,
    delay: float,
) -> None:
    expected = _expected_hashes(sums, artifact_root)
    last_error: Exception | None = None
    urls: list[str] = []
    for attempt in range(1, attempts + 1):
        try:
            urls = _verify_metadata(project=project, version=version, expected=expected)
            _verify_integrity_metadata(
                project=project,
                version=version,
                filenames=sorted(expected),
            )
            break
        except (RuntimeError, urllib.error.HTTPError, urllib.error.URLError) as exc:
            last_error = exc
            if attempt == attempts:
                raise RuntimeError(
                    f"PyPI release did not become verifiable after {attempts} attempts: {exc}"
                ) from exc
            print(f"PyPI verification attempt {attempt}/{attempts} is not ready: {exc}")
            time.sleep(delay)
    if not urls:
        raise RuntimeError(f"PyPI verification produced no distribution URLs: {last_error}")

    verifier = shutil.which("pypi-attestations")
    if verifier is None:
        raise RuntimeError("pypi-attestations is required for cryptographic provenance checks")
    for url in urls:
        subprocess.run(
            [verifier, "verify", "pypi", "--repository", repository, url],
            check=True,
        )
    print(f"verified PyPI {project} {version}: {', '.join(sorted(expected))}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--sums", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--attempts", type=int, default=12)
    parser.add_argument("--delay", type=float, default=10.0)
    args = parser.parse_args()
    if args.attempts < 1:
        parser.error("--attempts must be at least 1")
    verify_release(
        project=args.project,
        version=args.version,
        sums=args.sums,
        artifact_root=args.artifact_root,
        repository=args.repository,
        attempts=args.attempts,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()
