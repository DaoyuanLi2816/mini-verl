"""Decide whether exact release-candidate bytes still need PyPI publication."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


def _request_version(project: str, version: str, *, attempts: int, timeout: int) -> Any | None:
    project_part = urllib.parse.quote(project, safe="")
    version_part = urllib.parse.quote(version, safe="")
    request = urllib.request.Request(
        f"https://pypi.org/pypi/{project_part}/{version_part}/json",
        headers={"Accept": "application/json", "User-Agent": "miniVERL-release-state"},
    )
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            if attempt == attempts:
                raise RuntimeError(f"PyPI JSON request failed with HTTP {exc.code}") from None
        except (urllib.error.URLError, TimeoutError):
            if attempt == attempts:
                raise RuntimeError("PyPI JSON request failed after bounded retries") from None
        time.sleep(min(attempt, 3))
    raise AssertionError("retry loop must return or raise")


def publish_needed(
    manifest_path: Path,
    *,
    project: str,
    attempts: int = 4,
    timeout: int = 15,
) -> bool:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("kind") != "miniverl_release_candidate":
        raise ValueError("candidate manifest kind is invalid")
    version = payload.get("miniverl_version")
    expected = {payload[key]["filename"]: payload[key]["sha256"] for key in ("wheel", "sdist")}
    public = _request_version(project, version, attempts=attempts, timeout=timeout)
    if public is None:
        return True
    urls = public.get("urls") if isinstance(public, dict) else None
    if not isinstance(urls, list):
        raise ValueError("PyPI response has no release file list")
    observed: dict[str, str] = {}
    for item in urls:
        filename = item.get("filename") if isinstance(item, dict) else None
        digest = (item.get("digests") or {}).get("sha256") if isinstance(item, dict) else None
        if not isinstance(filename, str) or not isinstance(digest, str):
            raise ValueError("PyPI release file identity is incomplete")
        if filename in observed:
            raise ValueError(f"PyPI release repeats file {filename!r}")
        observed[filename] = digest
    if observed != expected:
        raise ValueError(
            "PyPI version exists but its complete file set does not match the candidate"
        )
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--project", default="miniverl")
    parser.add_argument("--attempts", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=15)
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args()
    needed = publish_needed(
        args.manifest,
        project=args.project,
        attempts=args.attempts,
        timeout=args.timeout,
    )
    result = {"publish_needed": needed}
    if args.github_output is not None:
        with args.github_output.open("a", encoding="utf-8", newline="\n") as output:
            output.write(f"publish_needed={str(needed).lower()}\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
