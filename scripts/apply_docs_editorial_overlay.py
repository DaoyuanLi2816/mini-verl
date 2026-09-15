"""Apply reviewed prose corrections to one immutable release's disposable docs tree.

The manifest binds the release commit and both versions of every page. It is
not a mechanism for serving arbitrary main documentation as stable. Record a
new editorial review explicitly with --record PATH [...]; changes to a recorded
page require refreshing its review hashes. A later release uses its own docs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path("scripts/docs_editorial_overlay.json")
RELEASE_COMMIT = "7a6ad7d9fa6e05787dca1b950ba6706c36384382"


def normalized_bytes(path: Path) -> bytes:
    """Git checks text out as CRLF on Windows and LF in deployment."""
    return path.read_bytes().replace(b"\r\n", b"\n")


def _allowed(path: str) -> bool:
    parts = PurePosixPath(path).parts
    return (
        bool(parts)
        and "\\" not in path
        and ".." not in parts
        and not PurePosixPath(path).is_absolute()
        and (path == "mkdocs.yml" or (parts[0] == "docs" and path.endswith(".md")))
    )


def _git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(root), *args])


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def apply_overlay(root: Path, stable: Path, selected_commit: str) -> int:
    manifest = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    if selected_commit != manifest["release_commit"]:
        return 0
    pending = []
    for relative, hashes in manifest["files"].items():
        if not _allowed(relative):
            raise ValueError(f"editorial overlay path refused: {relative}")
        source, destination = root / relative, stable / relative
        # Only repository-contained regular text files can be overlaid.
        for path, directory in ((source, root), (destination, stable)):
            if not path.resolve().is_relative_to(directory.resolve()) or not path.is_file():
                raise ValueError(f"editorial overlay path refused: {relative}")
        after = normalized_bytes(source)
        if _digest(normalized_bytes(destination)) != hashes["before"]:
            raise ValueError(f"stable editorial source differs: {relative}")
        if _digest(after) != hashes["after"]:
            raise ValueError(f"reviewed editorial replacement differs: {relative}")
        pending.append((destination, after))
    # Validate every page first; a mismatch must leave the checkout untouched.
    for destination, content in pending:
        destination.write_bytes(content)
    return len(pending)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--stable-source", type=Path)
    mode.add_argument("--record", nargs="+")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.record:
        files = {}
        for relative in sorted(args.record):
            if not _allowed(relative):
                parser.error(f"editorial overlay path refused: {relative}")
            before = _git(ROOT, "show", f"{RELEASE_COMMIT}:{relative}").replace(b"\r\n", b"\n")
            files[relative] = {
                "before": _digest(before),
                "after": _digest(normalized_bytes(ROOT / relative)),
            }
        (ROOT / MANIFEST).write_text(
            json.dumps({"release_commit": RELEASE_COMMIT, "files": files}, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    elif args.check:
        import yaml

        manifest = json.loads((ROOT / MANIFEST).read_text(encoding="utf-8"))
        state = yaml.safe_load((ROOT / "release-state.yaml").read_text(encoding="utf-8"))
        if state["stable"]["release_commit"] != manifest["release_commit"]:
            print("editorial overlay is inactive for this release")
            return
        for relative, hashes in manifest["files"].items():
            if not _allowed(relative):
                parser.error(f"editorial overlay path refused: {relative}")
            if _digest(normalized_bytes(ROOT / relative)) != hashes["after"]:
                parser.error(f"editorial review needs refreshing: {relative}")
        print("reviewed editorial replacements agree")
    else:
        commit = _git(args.stable_source, "rev-parse", "HEAD").decode().strip()
        count = apply_overlay(ROOT, args.stable_source, commit)
        print(f"applied {count} release-bound editorial corrections")


if __name__ == "__main__":
    main()
