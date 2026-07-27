"""Run directory layout, manifests and JSONL event/metric logs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from miniverl.errors import RunNotFoundError

__all__ = [
    "RunPaths",
    "make_run_id",
    "utc_now",
    "JsonlWriter",
    "read_jsonl",
    "canonical_json",
    "write_text",
    "write_json",
    "read_json",
]

_SAFE = re.compile(r"[^a-zA-Z0-9._-]+")


def utc_now() -> str:
    """Current UTC timestamp, second resolution, ISO-8601."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def make_run_id(name: str, *, explicit: str | None = None) -> str:
    """Build a filesystem-safe run id."""
    if explicit:
        return _SAFE.sub("-", explicit).strip("-") or "run"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    slug = _SAFE.sub("-", name).strip("-").lower() or "run"
    return f"{stamp}-{slug}"


@dataclass(frozen=True)
class RunPaths:
    """Canonical layout of a miniVERL run directory."""

    root: Path

    @classmethod
    def create(cls, output_dir: str | Path, run_id: str) -> RunPaths:
        """Create the directory tree for a new run."""
        root = Path(output_dir) / run_id
        root.mkdir(parents=True, exist_ok=True)
        paths = cls(root)
        paths.teacher_cache.mkdir(parents=True, exist_ok=True)
        paths.checkpoints.mkdir(parents=True, exist_ok=True)
        return paths

    @classmethod
    def open(cls, root: str | Path) -> RunPaths:
        """Open an existing run directory."""
        path = Path(root)
        if not path.is_dir():
            raise RunNotFoundError(
                f"run directory not found: {path}",
                hint="pass the path printed by `miniverl train`, e.g. runs/<run-id>",
            )
        if not (path / "manifest.json").is_file():
            raise RunNotFoundError(
                f"{path} does not look like a miniVERL run (no manifest.json)",
                hint="point at the run directory itself, not its parent",
            )
        return cls(path)

    # -- files -------------------------------------------------------------

    @property
    def config_original(self) -> Path:
        """The recipe exactly as the user wrote it."""
        return self.root / "config.original.yaml"

    @property
    def config_resolved(self) -> Path:
        """The recipe after every ``auto`` was resolved."""
        return self.root / "config.resolved.yaml"

    @property
    def manifest(self) -> Path:
        """Run identity, hardware and provenance."""
        return self.root / "manifest.json"

    @property
    def environment(self) -> Path:
        """Machine and package description."""
        return self.root / "environment.json"

    @property
    def metrics(self) -> Path:
        """One JSON object per logged step."""
        return self.root / "metrics.jsonl"

    @property
    def events(self) -> Path:
        """One JSON object per lifecycle event."""
        return self.root / "events.jsonl"

    @property
    def trajectories(self) -> Path:
        """All rollouts written during the run."""
        return self.root / "trajectories.jsonl"

    @property
    def eval_trajectories(self) -> Path:
        """Rollouts produced by evaluation passes."""
        return self.root / "eval_trajectories.jsonl"

    @property
    def teacher_cache(self) -> Path:
        """Teacher-target cache directory."""
        return self.root / "teacher-cache"

    @property
    def checkpoints(self) -> Path:
        """Checkpoint directory."""
        return self.root / "checkpoints"

    @property
    def eval_json(self) -> Path:
        """Final evaluation summary."""
        return self.root / "eval.json"

    @property
    def report_html(self) -> Path:
        """Default report location."""
        return self.root / "report.html"

    @property
    def summary_md(self) -> Path:
        """Markdown summary."""
        return self.root / "summary.md"

    @property
    def benchmark_json(self) -> Path:
        """Benchmark comparison output."""
        return self.root / "benchmark.json"


class JsonlWriter:
    """Append-only JSONL writer that flushes every record."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._count = 0

    def write(self, record: dict[str, Any]) -> None:
        """Append one record."""
        with self.path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=_fallback))
            fh.write("\n")
        self._count += 1

    @property
    def count(self) -> int:
        """Records written by this writer instance."""
        return self._count


def _fallback(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "value"):
        return obj.value
    return str(obj)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dicts."""
    p = Path(path)
    if not p.is_file():
        return []
    out: list[dict[str, Any]] = []
    for lineno, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        try:
            out.append(json.loads(text))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{p}:{lineno} is not valid JSON: {exc}") from exc
    return out


def canonical_json(payload: Any) -> str:
    """Serialize ``payload`` to the one JSON form this project writes to disk.

    Sorted keys and a trailing newline make the output a function of the data
    alone, so a file regenerated on another machine -- or through another code
    path -- is byte-identical. Anything that is diffed or checksummed must go
    through here rather than calling :func:`json.dumps` with its own options.
    """
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=_fallback) + "\n"
    )


def write_text(path: str | Path, text: str) -> Path:
    """Write ``text`` as UTF-8 with LF line endings on every platform.

    ``Path.write_text`` translates ``\\n`` to the platform separator, so the same
    run on Windows and Linux produces byte-different artifacts -- including the
    cache index and the checkpoint state, which carry checksums. Every artifact
    this project writes goes through here so that a run directory is a function
    of the computation and not of the operating system.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    return p


def write_json(path: str | Path, payload: Any) -> Path:
    """Write pretty-printed JSON in the canonical form."""
    return write_text(path, canonical_json(payload))


def read_json(path: str | Path) -> Any:
    """Read a JSON file."""
    p = Path(path)
    if not p.is_file():
        raise RunNotFoundError(f"expected JSON file not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))
