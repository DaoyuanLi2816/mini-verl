"""One canonical source for every public stable/development version claim.

After v0.6.2 was published and main advanced to 0.6.3.dev0, the repository still
told four different stories: the README said stable was v0.6.1, the docs
selector offered "Stable 0.6.1 / Development 0.6.2.dev0", and the quality record
paired ``release: 0.6.2`` with ``quality_floor: ... at v0.6.1``. Every one of
those was a hand-edited literal that a release simply forgot.

``release-state.yaml`` now owns the answer and this module projects it onto the
files that have to agree. ``--check`` is the gate; ``--write`` regenerates.

Run:

    python scripts/release_state.py --check
    python scripts/release_state.py --write
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

__all__ = [
    "ReleaseState",
    "apply_release_state",
    "check_release_state",
    "load_release_state",
    "rules_for",
]

RELEASE_STATE_FILE = "release-state.yaml"
_REPOSITORY = "https://github.com/DaoyuanLi2816/mini-verl"
_DEV_SUFFIX = re.compile(r"\.dev\d+$")


#: A release commit cannot name its own merge SHA, so the release phase may say
#: this instead. The post-release state-sync fills in the real value.
PENDING_COMMIT = "pending"

#: ``development``: main is building the next release, so the tree's version is a
#: ``.devN`` and differs from what is published. ``release``: this tree *is* the
#: release being cut, so its version equals the version being published. v0.6.2
#: had no such distinction, which is how its tag shipped a docs selector still
#: advertising "Stable 0.6.1 / Development 0.6.2.dev0".
PHASES = ("development", "release")


@dataclass(frozen=True)
class ReleaseState:
    """The published release and the version this tree is building."""

    stable_version: str
    stable_tag: str
    stable_commit: str
    stable_released_at: str
    development_version: str
    phase: str = "development"

    @property
    def preparing_version(self) -> str:
        """The release the development version will become (0.6.3.dev0 -> 0.6.3)."""
        return _DEV_SUFFIX.sub("", self.development_version)

    @property
    def is_release(self) -> bool:
        return self.phase == "release"

    def validate(self) -> None:
        if self.phase not in PHASES:
            raise ValueError(f"phase must be one of {', '.join(PHASES)}; got {self.phase!r}")
        if self.stable_tag != f"v{self.stable_version}":
            raise ValueError(f"stable.tag {self.stable_tag!r} must be v{self.stable_version}")
        if self.is_release:
            if self.development_version != self.stable_version:
                raise ValueError(
                    f"in the release phase development.version "
                    f"{self.development_version!r} must equal the version being published "
                    f"{self.stable_version!r}"
                )
            if self.stable_commit != PENDING_COMMIT and not re.fullmatch(
                r"[0-9a-f]{40}", self.stable_commit
            ):
                raise ValueError(
                    f"stable.release_commit must be a full 40-character SHA-1 or "
                    f"{PENDING_COMMIT!r} until the merge commit exists"
                )
        else:
            if not _DEV_SUFFIX.search(self.development_version):
                raise ValueError(
                    f"development.version {self.development_version!r} must end in .devN; "
                    "main is never a released version"
                )
            if self.preparing_version == self.stable_version:
                raise ValueError(
                    f"development.version {self.development_version!r} would re-release the "
                    f"published version {self.stable_version}"
                )
            if not re.fullmatch(r"[0-9a-f]{40}", self.stable_commit):
                raise ValueError("stable.release_commit must be a full 40-character SHA-1")
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", self.stable_released_at):
            raise ValueError("stable.released_at must be an ISO date")


def load_release_state(root: Path) -> ReleaseState:
    """Read and validate the canonical release state."""
    payload = yaml.safe_load((root / RELEASE_STATE_FILE).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError(f"{RELEASE_STATE_FILE} must be a schema_version 1 mapping")
    stable = payload.get("stable") or {}
    development = payload.get("development") or {}
    state = ReleaseState(
        stable_version=str(stable.get("version", "")),
        stable_tag=str(stable.get("tag", "")),
        stable_commit=str(stable.get("release_commit", "")),
        stable_released_at=str(stable.get("released_at", "")),
        development_version=str(development.get("version", "")),
        phase=str(payload.get("phase", "development")),
    )
    state.validate()
    return state


# --------------------------------------------------------------------- rules


@dataclass(frozen=True)
class Rule:
    """One projection of the release state onto a tracked file."""

    path: str
    description: str
    pattern: re.Pattern[str]
    expected: str
    #: ``False`` when the file is generated by another script and this module
    #: must only verify it rather than edit it.
    writable: bool = True

    def matches(self, text: str) -> list[re.Match[str]]:
        return list(self.pattern.finditer(text))


@dataclass(frozen=True)
class Presence:
    """A literal that must appear somewhere in a tracked file."""

    path: str
    description: str
    required: str
    remedy: str


def rules_for(state: ReleaseState) -> tuple[list[Rule], list[Presence]]:
    """Every file-level claim derived from ``state``."""
    rules = [
        Rule(
            path="src/miniverl/__init__.py",
            description="package __version__",
            pattern=re.compile(r'(?m)^__version__ = "(?P<value>[^"]+)"$'),
            expected=state.development_version,
        ),
        Rule(
            path="README.md",
            description="English stable/development statement",
            pattern=re.compile(r"PyPI `v(?P<value>[0-9][^`]*)` is stable"),
            expected=state.stable_version,
        ),
        Rule(
            path="README.zh-CN.md",
            description="Chinese stable/development statement",
            pattern=re.compile(r"PyPI `v(?P<value>[0-9][^`]*)` 是稳定版"),
            expected=state.stable_version,
        ),
        Rule(
            path="PYPI.md",
            description="PyPI long-description stable statement",
            pattern=re.compile(r"PyPI `v(?P<value>[0-9][^`]*)` is stable"),
            expected=state.stable_version,
            # Also generated by scripts/build_pypi_readme.py from README.md.
            # Both derive from the same README sentence, so writing here and
            # regenerating there converge; build_pypi_readme --check stays the
            # authority on the rest of the file.
        ),
        Rule(
            path="docs/overrides/main.html",
            description="docs channel stable attribute",
            pattern=re.compile(r'data-stable-version="(?P<value>[^"]*)"'),
            expected=state.stable_version,
        ),
        Rule(
            path="docs/overrides/main.html",
            description="docs channel development attribute",
            pattern=re.compile(r'data-dev-version="(?P<value>[^"]*)"'),
            expected=state.development_version,
        ),
        Rule(
            path="docs/overrides/main.html",
            description="docs version selector stable label",
            pattern=re.compile(r'<option value="stable">Stable (?P<value>[^<]*)</option>'),
            expected=state.stable_version,
        ),
        Rule(
            path="docs/overrides/main.html",
            description="docs version selector development label",
            pattern=re.compile(r'<option value="dev">Development (?P<value>[^<]*)</option>'),
            expected=state.development_version,
        ),
        Rule(
            path="CITATION.cff",
            description="citation version",
            pattern=re.compile(r"(?m)^version: (?P<value>.+)$"),
            expected=state.stable_version,
        ),
        Rule(
            path="CITATION.cff",
            description="citation release date",
            pattern=re.compile(r"(?m)^date-released: (?P<value>.+)$"),
            expected=state.stable_released_at,
        ),
        Rule(
            path="SECURITY.md",
            description="supported stable line",
            pattern=re.compile(r"The current supported stable line is `(?P<value>[^`]+)`"),
            expected=state.stable_version,
        ),
        Rule(
            path="PROJECT_STATE.md",
            description="current development product line",
            pattern=re.compile(r"Development `(?P<value>[^`]+)` has a closed typed profile"),
            expected=state.development_version,
        ),
        Rule(
            path="CHANGELOG.md",
            description="Unreleased comparison link",
            pattern=re.compile(
                r"(?m)^\[Unreleased\]: "
                + re.escape(_REPOSITORY)
                + r"/compare/v(?P<value>[^.]+\.[^.]+\.[^.]+)\.\.\.HEAD$"
            ),
            expected=state.stable_version,
        ),
    ]
    presence = [
        Presence(
            path="docs/release-checklist.md",
            description="a section for the release being prepared",
            required=f"## v{state.preparing_version}",
            remedy=(
                f"add a '## v{state.preparing_version} ...' section describing the "
                "gates for the release main is building"
            ),
        ),
        Presence(
            path="PROJECT_STATE.md",
            description="the canonical stable-release line",
            required=(
                f"Canonical release state: releasing `v{state.stable_version}`."
                if state.is_release
                else f"Canonical release state: stable `v{state.stable_version}` "
                f"(`{state.stable_commit}`), development `{state.development_version}`."
            ),
            remedy="copy the line from release-state.yaml into the PROJECT_STATE header",
        ),
    ]
    return rules, presence


# ------------------------------------------------------------------ quality


def _check_quality_record(root: Path, state: ReleaseState) -> list[str]:
    """The measurement record must not name two different releases."""
    path = root / "docs" / "generated" / "quality.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"docs/generated/quality.json: cannot read ({exc})"]
    problems: list[str] = []
    release = str(payload.get("release", ""))
    allowed = {state.stable_version, state.preparing_version}
    if release not in allowed:
        problems.append(
            f"docs/generated/quality.json: release {release!r} is neither the published "
            f"version {state.stable_version} nor the one being prepared "
            f"{state.preparing_version}"
        )
    floor = str(payload.get("quality_floor", ""))
    versions = set(re.findall(r"v(\d+\.\d+\.\d+)", floor))
    if versions and versions != {release}:
        problems.append(
            f"docs/generated/quality.json: quality_floor names {sorted(versions)} but the "
            f"record is for release {release!r}"
        )
    return problems


# ------------------------------------------------------------ check / write


def check_release_state(root: Path, state: ReleaseState | None = None) -> list[str]:
    """Return every disagreement with the canonical release state."""
    state = state or load_release_state(root)
    rules, presence = rules_for(state)
    problems: list[str] = []
    for rule in rules:
        path = root / rule.path
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            problems.append(f"{rule.path}: cannot read ({exc})")
            continue
        found = rule.matches(text)
        if not found:
            problems.append(f"{rule.path}: no {rule.description} to check")
            continue
        for match in found:
            actual = match.group("value")
            if actual != rule.expected:
                problems.append(
                    f"{rule.path}: {rule.description} is {actual!r}, expected {rule.expected!r}"
                )
    for item in presence:
        path = root / item.path
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            problems.append(f"{item.path}: cannot read ({exc})")
            continue
        if item.required not in text:
            problems.append(f"{item.path}: missing {item.description}; {item.remedy}")
    problems.extend(_check_quality_record(root, state))
    return problems


def apply_release_state(root: Path, state: ReleaseState | None = None) -> list[str]:
    """Rewrite every writable projection. Returns the paths that changed."""
    state = state or load_release_state(root)
    rules, _ = rules_for(state)
    changed: list[str] = []
    for rule in rules:
        if not rule.writable:
            continue
        path = root / rule.path
        text = path.read_text(encoding="utf-8")

        def _replace(match: re.Match[str], rule: Rule = rule) -> str:
            start, end = match.span("value")
            return (
                match.group(0)[: start - match.start()]
                + rule.expected
                + match.group(0)[end - match.start() :]
            )

        updated = rule.pattern.sub(_replace, text)
        if updated != text:
            path.write_text(updated, encoding="utf-8", newline="")
            changed.append(rule.path)
    return sorted(set(changed))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check", action="store_true", help="Fail on any disagreement.")
    group.add_argument("--write", action="store_true", help="Regenerate writable claims.")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    root = arguments.root
    state = load_release_state(root)
    if arguments.write:
        changed = apply_release_state(root, state)
        for path in changed:
            print(f"updated {path}")
        if not changed:
            print("no writable claim needed updating")
    problems = check_release_state(root, state)
    if problems:
        print(
            f"release state disagreement ({len(problems)}); "
            f"canonical source is {RELEASE_STATE_FILE}:",
            file=sys.stderr,
        )
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print(
        f"release state agrees: stable {state.stable_tag}, development {state.development_version}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
