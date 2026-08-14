"""Check repository-local Markdown files, links, images, and heading anchors."""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path
from urllib.parse import unquote

_MARKDOWN_TARGET = re.compile(r"!?\[[^\]]*]\(([^)\s]+)")
_HTML_TARGET = re.compile(r'\b(?:src|href)="([^"]+)"')
_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*#*\s*$")
_EXPLICIT_ANCHOR = re.compile(r'\b(?:id|name)="([^"]+)"')
_FENCE = re.compile(r"^\s*(```|~~~)")


def _active_lines(path: Path) -> list[str]:
    active: list[str] = []
    fence: str | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        marker = _FENCE.match(line)
        if marker:
            current = marker.group(1)
            if fence is None:
                fence = current
            elif current == fence:
                fence = None
            continue
        if fence is None:
            active.append(line)
    return active


def _slug(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\[([^\]]+)]\([^)]+\)", r"\1", text)
    text = text.replace("`", "").replace("*", "").replace("~", "")
    kept = []
    for character in text.casefold().strip():
        category = unicodedata.category(character)
        if category[0] in {"L", "M", "N"} or character in {" ", "-", "_"}:
            kept.append(character)
    return re.sub(r"\s", "-", "".join(kept))


def _anchors(path: Path) -> set[str]:
    anchors: set[str] = set()
    seen: dict[str, int] = {}
    for line in _active_lines(path):
        anchors.update(_EXPLICIT_ANCHOR.findall(line))
        match = _HEADING.match(line)
        if match is None:
            continue
        base = _slug(match.group(1))
        duplicate = seen.get(base, 0)
        seen[base] = duplicate + 1
        anchors.add(base if duplicate == 0 else f"{base}-{duplicate}")
    return anchors


def check_markdown_links(root: Path) -> list[str]:
    """Return every broken repository-local target."""
    ignored_parts = {
        ".artifacts",
        ".git",
        ".venv",
        "dist",
        "build",
        "history",
        "runs",
        "site",
    }
    files = [
        path
        for path in root.rglob("*.md")
        if not ignored_parts.intersection(path.relative_to(root).parts)
    ]
    anchor_cache: dict[Path, set[str]] = {}
    problems: list[str] = []
    for source in sorted(files):
        lines = _active_lines(source)
        for number, line in enumerate(lines, start=1):
            targets = [
                *((target, False) for target in _MARKDOWN_TARGET.findall(line)),
                *((target, True) for target in _HTML_TARGET.findall(line)),
            ]
            for raw_target, is_raw_html in targets:
                if raw_target.startswith(("http://", "https://", "mailto:", "data:")):
                    continue
                if raw_target.startswith("#"):
                    target_path = source
                    fragment = raw_target[1:]
                else:
                    path_text, separator, fragment = raw_target.partition("#")
                    relative_base = source.parent
                    if (
                        is_raw_html
                        and source.suffix.lower() == ".md"
                        and (root.resolve() / "docs") in source.parents
                    ):
                        relative_base = (
                            source.parent
                            if source.name.lower() == "index.md"
                            else source.parent / source.stem
                        )
                    target_path = (relative_base / unquote(path_text)).resolve()
                    if not separator:
                        fragment = ""
                try:
                    target_path.relative_to(root.resolve())
                except ValueError:
                    problems.append(
                        f"{source.relative_to(root)}:{number}: target leaves repository: {raw_target}"
                    )
                    continue
                if not target_path.exists():
                    problems.append(
                        f"{source.relative_to(root)}:{number}: missing target: {raw_target}"
                    )
                    continue
                if fragment and target_path.is_file() and target_path.suffix.lower() == ".md":
                    anchors = anchor_cache.setdefault(target_path, _anchors(target_path))
                    decoded = unquote(fragment).casefold()
                    if decoded not in anchors:
                        problems.append(
                            f"{source.relative_to(root)}:{number}: missing anchor "
                            f"#{fragment} in {target_path.relative_to(root)}"
                        )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    problems = check_markdown_links(args.root.resolve())
    if problems:
        parser.error("\n".join(problems))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
