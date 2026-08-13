"""Build the PyPI long description with navigable, release-stable links."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

OWNER = "DaoyuanLi2816"
REPOSITORY = "mini-verl"
GITHUB_ROOT = f"https://github.com/{OWNER}/{REPOSITORY}"
RAW_ROOT = f"https://raw.githubusercontent.com/{OWNER}/{REPOSITORY}"

_LINKED_IMAGE = re.compile(
    r"\[(?P<label>!\[[^\]]*])"
    r"\((?P<image_target>[^)\s]+)(?P<image_trailer>[^)]*)\)]"
    r"\((?P<link_target>[^)\s]+)(?P<link_trailer>[^)]*)\)"
)
_MARKDOWN_LINK = re.compile(r"(!?\[[^\]]*])\(([^)\s]+)([^)]*)\)")
_ANY_MARKDOWN_TARGET = re.compile(r"]\((?P<target>[^)\s]+)[^)]*\)")
_HTML_TARGET = re.compile(r'(?P<prefix>\b(?:src|srcset|href)=")(?P<target>[^"]+)(?P<suffix>")')
_VERSION = re.compile(r'^__version__\s*=\s*"(?P<version>[^"]+)"\s*$', re.MULTILINE)


def project_version(root: Path) -> str:
    """Return the version declared by the package."""
    source = (root / "src" / "miniverl" / "__init__.py").read_text(encoding="utf-8")
    match = _VERSION.search(source)
    if match is None:  # pragma: no cover - repository corruption
        raise ValueError("src/miniverl/__init__.py does not declare __version__")
    return match.group("version")


def release_ref(version: str) -> str:
    """Use a tag for releases and ``main`` for development snapshots."""
    return "main" if ".dev" in version else f"v{version}"


def _is_passthrough(target: str) -> bool:
    return target.startswith(("https://", "http://", "#", "mailto:", "data:"))


def _project_target(target: str, *, ref: str, image: bool) -> str:
    """Convert one README target into a navigable repository URL."""
    if target.startswith(f"{GITHUB_ROOT}/blob/"):
        remainder = target.removeprefix(f"{GITHUB_ROOT}/blob/")
        _old_ref, separator, path = remainder.partition("/")
        return f"{GITHUB_ROOT}/blob/{ref}/{path}" if separator else target
    if target.startswith(f"{RAW_ROOT}/"):
        remainder = target.removeprefix(f"{RAW_ROOT}/")
        _old_ref, separator, path = remainder.partition("/")
        return f"{RAW_ROOT}/{ref}/{path}" if separator else target
    if _is_passthrough(target):
        return target

    path, fragment_separator, fragment = target.partition("#")
    normalized = path.removeprefix("./")
    base = RAW_ROOT if image else f"{GITHUB_ROOT}/blob"
    converted = f"{base}/{ref}/{normalized}"
    if fragment_separator:
        converted += f"#{fragment}"
    return converted


def relative_project_targets(text: str) -> list[str]:
    """Return every Markdown/HTML target that is not public or intra-document."""
    targets = [match.group("target") for match in _ANY_MARKDOWN_TARGET.finditer(text)]
    targets.extend(match.group("target") for match in _HTML_TARGET.finditer(text))
    return [target for target in targets if not _is_passthrough(target)]


def build_pypi_readme(root: Path, *, version: str | None = None) -> str:
    """Render ``README.md`` with absolute URLs pinned to the appropriate ref."""
    selected_version = version or project_version(root)
    ref = release_ref(selected_version)
    source = (root / "README.md").read_text(encoding="utf-8")

    def replace_linked_image(match: re.Match[str]) -> str:
        image_target = _project_target(
            match.group("image_target"),
            ref=ref,
            image=True,
        )
        link_target = _project_target(
            match.group("link_target"),
            ref=ref,
            image=False,
        )
        return (
            f"[{match.group('label')}({image_target}{match.group('image_trailer')})]"
            f"({link_target}{match.group('link_trailer')})"
        )

    source = _LINKED_IMAGE.sub(replace_linked_image, source)

    def replace_markdown(match: re.Match[str]) -> str:
        label, target, trailer = match.groups()
        converted = _project_target(target, ref=ref, image=label.startswith("!"))
        return f"{label}({converted}{trailer})"

    rendered = _MARKDOWN_LINK.sub(replace_markdown, source)

    def replace_html(match: re.Match[str]) -> str:
        target = match.group("target")
        image = match.group("prefix").startswith(("src=", "srcset="))
        converted = _project_target(target, ref=ref, image=image)
        return f"{match.group('prefix')}{converted}{match.group('suffix')}"

    rendered = _HTML_TARGET.sub(replace_html, rendered)
    relative = relative_project_targets(rendered)
    if relative:
        targets = ", ".join(sorted(set(relative)))
        raise ValueError(f"PyPI description contains relative project targets: {targets}")
    return rendered


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    destination = root / "PYPI.md"
    rendered = build_pypi_readme(root)
    if args.check:
        if not destination.is_file() or destination.read_text(encoding="utf-8") != rendered:
            parser.error("PYPI.md is stale; run scripts/build_pypi_readme.py")
        return 0
    destination.write_text(rendered, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
