"""Fail on mojibake and unintended byte-order marks in tracked text files.

`docs/release-checklist.md` shipped `鈥?not applicable` -- a UTF-8 en dash that
was decoded as GBK and re-encoded, which is what happens when a file is written
through a non-UTF-8 console. The bytes are valid UTF-8 afterwards, so no
encoding error is ever raised and the damage survives review.

The check is deliberately targeted rather than exhaustive. A blanket "no CJK
outside the Chinese README" rule would fire on the Unicode fixtures in the test
suite and on the `中文` README link, so it would be switched off within a week.
Instead it looks for three unambiguous signals:

* the leading characters this specific mis-decoding produces -- UTF-8 `—`, `→`
  and curly quotes read as GBK all begin `鈥`, `鈫`, `锛`, `銆`;
* U+FFFD REPLACEMENT CHARACTER, which is always damage;
* a UTF-8 byte-order mark. Nothing here reads UTF-8-with-BOM deliberately, and
  it breaks shebangs, YAML front matter and strict JSON parsers.

This will not catch every possible encoding accident, only the class that has
actually occurred. Widen `MOJIBAKE_LEADERS` when a new one appears.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import unicodedata
from pathlib import Path

#: Leading characters produced when UTF-8 punctuation is decoded as GBK.
#: `鈥` is a mangled em dash, `鈫` a mangled arrow, `锛`/`銆` mangled CJK
#: punctuation. None of these is ever written on purpose here, including in the
#: Chinese README, so this needs no per-file allowlist.
MOJIBAKE_LEADERS = ("鈥", "鈫", "锛", "銆", "鎴", "鑰", "娴", "鏂")

#: Suffixes worth reading as text. Frozen scientific results are included on
#: purpose: mojibake in a published result would be a real defect. Nothing here
#: rewrites them -- the check only reports.
TEXT_SUFFIXES = frozenset(
    {
        ".py",
        ".md",
        ".txt",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".cfg",
        ".ini",
        ".sh",
        ".ps1",
        ".html",
        ".css",
        ".js",
        ".svg",
        ".cff",
        ".jsonl",
    }
)

_BOM = "﻿"
_REPLACEMENT = "�"


def _tracked_files(root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        capture_output=True,
        check=True,
        text=True,
    )
    return [root / name for name in result.stdout.split("\0") if name]


def _describe(text: str, index: int) -> str:
    """Locate a suspicious character without echoing the surrounding content."""
    line = text.count("\n", 0, index) + 1
    character = text[index]
    try:
        name = unicodedata.name(character)
    except ValueError:
        name = "unnamed"
    return f"line {line}: U+{ord(character):04X} {name}"


def check_text_integrity(root: Path) -> list[str]:
    """Return one problem string per damaged file."""
    problems: list[str] = []
    for path in _tracked_files(root):
        if path.suffix.lower() not in TEXT_SUFFIXES or not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        try:
            raw = path.read_bytes()
        except OSError:
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            problems.append(f"{relative}: not valid UTF-8 ({exc.reason})")
            continue

        if text.startswith(_BOM):
            problems.append(f"{relative}: starts with a UTF-8 byte-order mark")

        replacement = text.find(_REPLACEMENT)
        if replacement != -1:
            problems.append(f"{relative}: {_describe(text, replacement)} (replacement character)")

        for leader in MOJIBAKE_LEADERS:
            index = text.find(leader)
            if index != -1:
                problems.append(
                    f"{relative}: {_describe(text, index)} "
                    "(mis-decoded UTF-8 punctuation; rewrite the file as UTF-8)"
                )
                break
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    problems = check_text_integrity(args.root)
    if problems:
        print("text integrity check failed:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("text integrity check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
