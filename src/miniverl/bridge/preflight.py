"""Structural preflight for an untrusted bundle tree.

Every other bridge check reads bundle content: ``SHA256SUMS``, reward source,
YAML, JSON, tokenizer files, Parquet, text metadata. Each of those reads follows
whatever the path resolves to. A bundle that ships a symlink named
``model/adapter_config.json`` pointing at ``C:/Users/<name>/.ssh/id_ed25519``
gets that file opened, hashed, and — for the text metadata scan — searched for
credential patterns, with the findings reported by path.

So the tree itself has to be validated before anything opens a file. This module
walks the bundle with ``lstat`` only, never following a link, and refuses:

* file and directory symlinks;
* Windows junctions and other reparse points;
* devices, sockets, FIFOs and anything else that is not a regular file;
* any entry whose resolved path escapes the bundle root;
* trees over a bounded file count or nominal byte total.

It proves the tree is a plain directory of regular files inside one root. It
proves nothing about the *content* of those files.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Any

__all__ = [
    "PREFLIGHT_MAX_FILES",
    "PREFLIGHT_MAX_NOMINAL_BYTES",
    "preflight_bundle_tree",
]

#: A real exported bundle is a few dozen files. These bound a hostile tree
#: without constraining a legitimate one.
PREFLIGHT_MAX_FILES = 10_000
PREFLIGHT_MAX_NOMINAL_BYTES = 8 * 1024 * 1024 * 1024
PREFLIGHT_MAX_DEPTH = 32

_FILE_ATTRIBUTE_REPARSE_POINT = 0x400


def _rejection(path: str, reason: str, detail: str) -> dict[str, str]:
    return {"path": path, "reason": reason, "detail": detail}


def _is_reparse_point(entry_stat: os.stat_result) -> bool:
    """Windows junctions are not symlinks but still redirect the path."""
    attributes = getattr(entry_stat, "st_file_attributes", 0)
    return bool(attributes & _FILE_ATTRIBUTE_REPARSE_POINT)


def _describe_mode(entry_stat: os.stat_result) -> str:
    mode = entry_stat.st_mode
    if stat.S_ISDIR(mode):
        return "directory"
    if stat.S_ISCHR(mode):
        return "character device"
    if stat.S_ISBLK(mode):
        return "block device"
    if stat.S_ISFIFO(mode):
        return "FIFO"
    if stat.S_ISSOCK(mode):
        return "socket"
    if stat.S_ISLNK(mode):
        return "symbolic link"
    return "non-regular file"


def preflight_bundle_tree(
    root: str | Path,
    *,
    max_files: int = PREFLIGHT_MAX_FILES,
    max_nominal_bytes: int = PREFLIGHT_MAX_NOMINAL_BYTES,
    max_depth: int = PREFLIGHT_MAX_DEPTH,
) -> dict[str, Any]:
    """Validate the bundle tree shape before any content is read.

    Fails closed: an unreadable directory, an unresolvable path or a stat error
    is a rejection, not a skipped entry.
    """
    bundle = Path(root)
    check: dict[str, Any] = {
        "status": "fail",
        "files": 0,
        "directories": 0,
        "nominal_bytes": 0,
        "rejections": [],
        "limits": {
            "max_files": int(max_files),
            "max_nominal_bytes": int(max_nominal_bytes),
            "max_depth": int(max_depth),
        },
        "scope": (
            "tree shape only: regular files inside one root, no link or reparse "
            "redirection. This says nothing about file content."
        ),
    }
    rejections: list[dict[str, str]] = check["rejections"]

    try:
        anchor = bundle.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        rejections.append(_rejection(str(bundle), "unresolvable_root", str(exc)))
        return check
    if not anchor.is_dir():
        rejections.append(
            _rejection(str(bundle), "root_not_a_directory", "bundle root must be a directory")
        )
        return check

    files = 0
    directories = 0
    nominal_bytes = 0
    # Explicit stack, so a deep or cyclic tree cannot exhaust the interpreter
    # stack the way a recursive walk would.
    stack: list[tuple[Path, int]] = [(anchor, 0)]
    while stack:
        current, depth = stack.pop()
        if depth > max_depth:
            rejections.append(
                _rejection(
                    _relative(current, anchor), "max_depth_exceeded", f"deeper than {max_depth}"
                )
            )
            return check
        try:
            with os.scandir(current) as scan:
                entries = sorted(scan, key=lambda item: item.name)
        except OSError as exc:
            rejections.append(
                _rejection(_relative(current, anchor), "unreadable_directory", str(exc))
            )
            return check
        for entry in entries:
            path = Path(entry.path)
            relative = _relative(path, anchor)
            try:
                entry_stat = entry.stat(follow_symlinks=False)
            except OSError as exc:
                rejections.append(_rejection(relative, "unstattable_entry", str(exc)))
                return check

            if entry.is_symlink() or stat.S_ISLNK(entry_stat.st_mode):
                rejections.append(
                    _rejection(
                        relative,
                        "symlink",
                        "a symbolic link can redirect a read outside the bundle",
                    )
                )
                return check
            if _is_reparse_point(entry_stat):
                rejections.append(
                    _rejection(
                        relative, "reparse_point", "junctions and reparse points redirect a read"
                    )
                )
                return check

            # The entry is not a link, so resolving it must stay under the root.
            try:
                resolved = path.resolve(strict=True)
            except (OSError, RuntimeError) as exc:
                rejections.append(_rejection(relative, "unresolvable_entry", str(exc)))
                return check
            if not _within(resolved, anchor):
                rejections.append(
                    _rejection(relative, "escapes_bundle_root", f"resolves to {resolved}")
                )
                return check

            if stat.S_ISDIR(entry_stat.st_mode):
                directories += 1
                stack.append((path, depth + 1))
                continue
            if not stat.S_ISREG(entry_stat.st_mode):
                rejections.append(
                    _rejection(relative, "non_regular_file", _describe_mode(entry_stat))
                )
                return check

            files += 1
            nominal_bytes += int(entry_stat.st_size)
            if files > max_files:
                rejections.append(
                    _rejection(relative, "max_files_exceeded", f"more than {max_files} files")
                )
                return check
            if nominal_bytes > max_nominal_bytes:
                rejections.append(
                    _rejection(
                        relative,
                        "max_nominal_bytes_exceeded",
                        f"nominal size exceeds {max_nominal_bytes} bytes",
                    )
                )
                return check

    check["files"] = files
    check["directories"] = directories
    check["nominal_bytes"] = nominal_bytes
    check["status"] = "ok"
    check["detail"] = (
        f"{files} regular file(s) in {directories} directory/directories under one root"
    )
    return check


def _relative(path: Path, anchor: Path) -> str:
    try:
        return path.relative_to(anchor).as_posix() or "."
    except ValueError:
        return str(path)


def _within(path: Path, anchor: Path) -> bool:
    try:
        path.relative_to(anchor)
    except ValueError:
        return False
    return True
