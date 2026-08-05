"""Exclusive, stem-scoped, transactional publication of a bridge output family.

A bridge invocation produces a *family* of files -- a recipe or a template plus
its report, or a Parquet file plus its sidecar and report -- that only mean
anything together. Publishing them one at a time lets a mid-run failure pair
invocation A's report with invocation B's artifact, and shared names such as
``import-report.json`` let two different ``--out`` stems silently overwrite each
other.

This module gives every invocation:

* a stem-specific target set, so ``foo.yaml`` only ever touches ``foo.*``;
* an exclusive per-stem reservation, so concurrent writers serialize;
* a fail-closed collision check that runs *before* anything is modified;
* a staging directory inside the destination, so the publish step is a
  same-filesystem rename rather than a copy;
* restore-on-failure, so a failure part-way through the rename phase puts the
  previous family back.

Scope of the guarantee: this is *transactional publication with in-process
rollback*, not multi-file crash atomicity. Every failure this process observes
-- an exception, a failed rename, a validation error -- is undone. A ``kill
-9``, a kernel panic or a power loss between two renames is not, and can leave
a mixed family on disk. Providing that would require publishing a versioned
directory and switching one pointer atomically, which is deliberately out of
scope; re-running the invocation with ``overwrite`` republishes a coherent set.
"""

from __future__ import annotations

import os
import re
import shutil
import uuid
from collections.abc import Iterable, Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any

from miniverl.errors import ConfigError, LifecycleError
from miniverl.utils.locking import RunLock
from miniverl.utils.runs import canonical_json, write_text

__all__ = [
    "DEFAULT_LOCK_TIMEOUT",
    "OutputCollisionError",
    "OutputTransaction",
    "SourceOutputAliasError",
    "dataset_output_targets",
    "import_output_targets",
    "reject_source_output_alias",
]

# A bounded wait, so two ordinary invocations against one stem serialize instead
# of failing. ``0`` keeps the historical non-blocking behaviour for callers that
# would rather fail immediately.
DEFAULT_LOCK_TIMEOUT = 30.0

_SAFE = re.compile(r"[^a-zA-Z0-9._-]+")


class OutputCollisionError(ConfigError):
    """One or more intended output paths already exist and were not replaced."""

    def __init__(self, conflicts: Iterable[Path]) -> None:
        self.conflicts = sorted({Path(path).resolve() for path in conflicts})
        listing = "\n".join(f"  - {path}" for path in self.conflicts)
        count = len(self.conflicts)
        super().__init__(
            f"refusing to replace {count} existing output path(s):\n{listing}",
            hint=(
                "pass --overwrite to replace this exact output family, or choose a "
                "different --out stem; nothing has been modified"
            ),
        )


class SourceOutputAliasError(ConfigError):
    """An input file is also one of the intended outputs. Nothing was modified."""

    def __init__(self, aliases: Mapping[str, tuple[Path, Path]]) -> None:
        self.aliases = dict(aliases)
        listing = "\n".join(
            f"  - {name}: {output}  aliases the input {source}"
            for name, (source, output) in sorted(self.aliases.items())
        )
        super().__init__(
            "source and output families must be distinct; choose a new --out path\n" + listing,
            hint=(
                "miniVERL has no in-place mode: --overwrite replaces a previous output "
                "family, it never authorizes destroying an input"
            ),
        )


def _normalize(path: Path) -> Path:
    """Absolute, symlink-resolved form that also works for absent paths."""
    try:
        return path.resolve(strict=False)
    except (OSError, RuntimeError):
        # resolve() can fail on pathological input; absolute+normalized is the
        # best remaining approximation and still catches ordinary aliases.
        return Path(os.path.normpath(str(Path.cwd() / path)))


def _same_file(left: Path, right: Path) -> bool:
    """Whether two paths denote one file, including link and case aliases."""
    try:
        # Authoritative when both exist: covers symlinks, hard links and
        # case-insensitive filesystems in one call.
        if left.exists() and right.exists():
            return left.samefile(right)
    except OSError:
        pass
    # One side does not exist yet, so fall back to normalized textual identity.
    # normcase folds case on Windows and is a no-op on POSIX.
    return os.path.normcase(str(_normalize(left))) == os.path.normcase(str(_normalize(right)))


def reject_source_output_alias(sources: Mapping[str, Path], targets: Mapping[str, Path]) -> None:
    """Fail before anything is reserved if an input is also an output.

    ``import-verl`` used to accept ``--out`` pointing at its own source config.
    With ``--overwrite`` a successful import replaced the user's input, and a
    rejected import *deleted* it while keeping the rejection report. There is no
    in-place mode, so any overlap is a hard error.
    """
    aliases: dict[str, tuple[Path, Path]] = {}
    for output_name, output in targets.items():
        for source_name, source in sources.items():
            if _same_file(Path(source), Path(output)):
                label = output_name if len(sources) == 1 else f"{output_name} <- {source_name}"
                aliases[label] = (Path(source), Path(output))
    if aliases:
        raise SourceOutputAliasError(aliases)


def import_output_targets(out: str | Path) -> dict[str, Path]:
    """Return the complete stem-specific family for ``miniverl import-verl``.

    One invocation publishes ``<stem>.yaml`` *or* ``<stem>.template.yaml``, never
    both, alongside exactly one ``<stem>.import-report.json``.
    """
    recipe = Path(out)
    stem = recipe.stem
    if not stem or stem in {".", ".."}:
        raise ConfigError(
            f"--out must name a recipe file, not {recipe}",
            hint="use --out recipes/<name>.yaml",
        )
    return {
        "recipe": recipe,
        "report": recipe.parent / f"{stem}.import-report.json",
        "template": recipe.parent / f"{stem}.template.yaml",
    }


def dataset_output_targets(out: str | Path) -> dict[str, Path]:
    """Return the complete family for ``miniverl convert-dataset``.

    The family is keyed on the requested Parquet path so ``train.parquet`` can
    only ever publish ``train.parquet``, ``train.parquet.miniverl.json`` and
    ``train.parquet.report.json``.
    """
    parquet = Path(out)
    if not parquet.name or parquet.name in {".", ".."}:
        raise ConfigError(
            f"--out must name a Parquet file, not {parquet}",
            hint="use --out <directory>/<name>.parquet",
        )
    return {
        "parquet": parquet,
        "sidecar": parquet.parent / f"{parquet.name}.miniverl.json",
        "report": parquet.parent / f"{parquet.name}.report.json",
    }


def _replace(source: Path, target: Path) -> None:
    """Publish one staged file. Seam kept module-level for fault injection."""
    source.replace(target)


class OutputTransaction:
    """Reserve, stage, validate and publish one output family.

    Rollback covers failures this process observes, not process death between
    two renames. See the module docstring for the exact scope.
    """

    def __init__(
        self,
        *,
        targets: Mapping[str, Path],
        stem: str,
        lock_root: str | Path,
        overwrite: bool = False,
        lock_timeout: float = DEFAULT_LOCK_TIMEOUT,
    ) -> None:
        self.targets: dict[str, Path] = {name: Path(path) for name, path in targets.items()}
        self.stem = stem
        self.directory = Path(lock_root)
        self.overwrite = bool(overwrite)
        self.staging: Path | None = None
        self._safe_stem = _SAFE.sub("-", stem).strip(".-") or "output"
        self._lock = RunLock(
            self.directory,
            f"bridge-output-{self._safe_stem}",
            timeout=float(lock_timeout),
        )
        self._staged: dict[str, Path] = {}
        self._discarded: set[str] = set()
        self._committed = False
        self._closed = False

    # ------------------------------------------------------------- lifecycle

    def begin(self) -> OutputTransaction:
        """Reserve the stem, refuse collisions, and open the staging directory."""
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock.acquire()
        try:
            conflicts = [path for path in self.targets.values() if path.exists()]
            if conflicts and not self.overwrite:
                raise OutputCollisionError(conflicts)
            staging = self.directory / f".{self._safe_stem}.{uuid.uuid4().hex}.staging"
            staging.mkdir(parents=True, exist_ok=False)
            self.staging = staging
        except BaseException:
            self._lock.release()
            self._closed = True
            raise
        return self

    def close(self) -> None:
        """Drop the staging directory and release the reservation. Idempotent."""
        if self._closed:
            return
        self._closed = True
        if self.staging is not None:
            shutil.rmtree(self.staging, ignore_errors=True)
        self._lock.release()

    def rollback(self) -> None:
        """Abandon an uncommitted transaction; published files are untouched."""
        self.close()

    def __enter__(self) -> OutputTransaction:
        return self.begin()

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        try:
            if exc_type is None and not self._committed:
                self.commit()
        finally:
            self.close()

    # ---------------------------------------------------------------- staging

    def path(self, name: str) -> Path:
        """Staging path for logical output ``name``."""
        if self.staging is None:
            raise LifecycleError("output transaction is not open; call begin() first")
        try:
            target = self.targets[name]
        except KeyError as exc:
            raise LifecycleError(f"{name!r} is not a planned output of this transaction") from exc
        return self.staging / target.name

    def write_bytes(self, name: str, data: bytes) -> None:
        """Stage exact bytes for ``name``."""
        path = self.path(name)
        path.write_bytes(data)
        self._staged[name] = path
        self._discarded.discard(name)

    def write_json(self, name: str, payload: Any) -> None:
        """Stage canonical machine JSON for ``name``."""
        path = self.path(name)
        write_text(path, canonical_json(payload))
        self._staged[name] = path
        self._discarded.discard(name)

    def claim(self, name: str) -> None:
        """Adopt a staged file written directly by an external writer."""
        path = self.path(name)
        if not path.is_file():
            raise LifecycleError(f"staged output {name!r} was never produced at {path}")
        self._staged[name] = path
        self._discarded.discard(name)

    def discard(self, name: str) -> None:
        """Declare that ``name`` must not exist once this family is published."""
        if name not in self.targets:
            raise LifecycleError(f"{name!r} is not a planned output of this transaction")
        self._staged.pop(name, None)
        self._discarded.add(name)

    # -------------------------------------------------------------- publishing

    @property
    def staged_paths(self) -> dict[str, Path]:
        """Logical name to staged file, for pre-publication validation."""
        return dict(self._staged)

    def commit(self) -> None:
        """Publish the staged family, restoring the previous one on failure.

        The restore path runs on any exception raised during the rename phase.
        It cannot run if the process is killed mid-phase.
        """
        if self.staging is None:
            raise LifecycleError("output transaction is not open; call begin() first")
        if self._committed:
            return
        backups = self.staging / ".replaced"
        backups.mkdir(exist_ok=True)
        saved: list[tuple[Path, Path]] = []
        published: list[Path] = []
        try:
            for name, target in self.targets.items():
                if name not in self._staged and name not in self._discarded:
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    keep = backups / f"{len(saved):02d}-{target.name}"
                    target.replace(keep)
                    saved.append((keep, target))
                if name in self._staged:
                    _replace(self._staged[name], target)
                    published.append(target)
            self._committed = True
        except BaseException:
            for target in published:
                with suppress(OSError):
                    target.unlink()
            for keep, target in saved:
                with suppress(OSError):
                    keep.replace(target)
            raise
