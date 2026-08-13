"""Verify that one PyPI release matches locally built distributions exactly."""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

_INTEGRITY_MEDIA_TYPE = "application/vnd.pypi.integrity.v1+json"


class _PinnedImageParser(HTMLParser):
    def __init__(self, target: str) -> None:
        super().__init__()
        self.target = target
        self.alts: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        if tag != "img":
            return
        attributes = dict(attrs)
        alt = attributes.get("alt")
        if attributes.get("src") == self.target and alt:
            self.alts.append(alt)


def _pinned_image_alts(description: str, target: str) -> list[str]:
    markdown_alts = re.findall(
        rf"!\[([^\]]+)\]\({re.escape(target)}\)",
        description,
    )
    parser = _PinnedImageParser(target)
    parser.feed(description)
    return [*markdown_alts, *parser.alts]


def _request_json(url: str, *, accept: str = "application/json") -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={"Accept": accept, "User-Agent": "miniVERL-release-verifier/0.2"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def _request_text(url: str) -> str:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "miniVERL-release-verifier/0.2"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8")


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


def _verify_long_description_links(
    *,
    project: str,
    version: str,
    repository: str,
    allow_rendered_page_challenge: bool = False,
) -> None:
    metadata = _request_json(f"https://pypi.org/pypi/{project}/{version}/json")
    description = str((metadata.get("info") or {}).get("description") or "")
    tag = f"v{version}"
    github = repository.rstrip("/")
    raw = github.replace("https://github.com/", "https://raw.githubusercontent.com/")
    required_paths = (
        "docs/single-gpu-guide.md",
        "CHANGELOG.md",
        "CITATION.cff",
        "LICENSE",
        "CONTRIBUTING.md",
        "SECURITY.md",
    )
    required_file_urls = [f"{github}/blob/{tag}/{path}" for path in required_paths]
    required_image_url = f"{raw}/{tag}/docs/banner.svg"
    required_urls = [*required_file_urls, required_image_url]
    missing = [url for url in required_urls if url not in description]
    if missing:
        raise RuntimeError(f"PyPI long description is missing release-pinned links: {missing}")
    if f"{github}/blob/main/" in description or f"{raw}/main/" in description:
        raise RuntimeError("stable PyPI long description still links repository files through main")

    image_matches = _pinned_image_alts(description, required_image_url)
    if not image_matches:
        raise RuntimeError(
            "PyPI long description does not use the release-pinned banner as an image"
        )
    project_links = sorted(
        set(
            re.findall(
                rf"https://(?:github\.com/{re.escape(github.removeprefix('https://github.com/'))}"
                rf"/blob|raw\.githubusercontent\.com/"
                rf"{re.escape(github.removeprefix('https://github.com/'))})/[^\s)\"<>]+",
                description,
            )
        )
    )
    if not project_links:
        raise RuntimeError("PyPI long description exposes no repository-file links")
    for url in project_links:
        _request_text(url)

    page = _request_text(f"https://pypi.org/project/{project}/{version}/")
    if "<title>Client Challenge</title>" in page:
        if allow_rendered_page_challenge:
            print(
                "warning: PyPI rendered project page returned a client challenge; "
                "release-pinned description links and their targets were verified through "
                "the public JSON API, but rendered-page inspection requires a browser"
            )
            return
        raise RuntimeError("PyPI rendered project page returned a client challenge")
    absent_from_page = [url for url in required_file_urls if url not in page]
    if absent_from_page:
        raise RuntimeError(f"rendered PyPI project page is missing links: {absent_from_page}")
    if "pypi-camo." not in page or not all(
        html.escape(alt, quote=True) in page for alt in image_matches
    ):
        raise RuntimeError("rendered PyPI project page is missing the proxied release banner")


def verify_release(
    *,
    project: str,
    version: str,
    sums: Path,
    artifact_root: Path,
    repository: str,
    attempts: int,
    delay: float,
    allow_rendered_page_challenge: bool,
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
            _verify_long_description_links(
                project=project,
                version=version,
                repository=repository,
                allow_rendered_page_challenge=allow_rendered_page_challenge,
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
    parser.add_argument(
        "--allow-rendered-page-challenge",
        action="store_true",
        help=(
            "accept PyPI's browser challenge only after public metadata and every "
            "release-pinned repository target have been verified"
        ),
    )
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
        allow_rendered_page_challenge=args.allow_rendered_page_challenge,
    )


if __name__ == "__main__":
    main()
