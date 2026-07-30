"""Cross-platform exclusive ownership for mutable run directories."""

from __future__ import annotations

import contextlib
import os
import re
from pathlib import Path
from typing import Any

from filelock import FileLock, Timeout

from miniverl.errors import LifecycleError, RunLockedError
from miniverl.utils.runs import utc_now, write_json_atomic

__all__ = ["RunLock"]

_SAFE = re.compile(r"[^a-zA-Z0-9._-]+")


class RunLock:
    """Hold one stable OS-backed lock outside the replaceable run directory."""

    def __init__(self, output_root: str | Path, run_id: str, *, timeout: float = 0.0) -> None:
        if timeout < 0:
            raise ValueError("run lock timeout must be non-negative")
        self.output_root = Path(output_root).resolve()
        self.run_id = run_id
        safe_id = _SAFE.sub("-", run_id).strip(".-") or "run"
        self.lock_root = self.output_root / ".miniverl-locks"
        self.path = self.lock_root / f"{safe_id}.lock"
        self.metadata_path = self.lock_root / f"{safe_id}.lock.json"
        self.timeout = float(timeout)
        self._lock = FileLock(str(self.path), timeout=self.timeout)
        self._acquired = False

    @property
    def acquired(self) -> bool:
        """Whether this object currently owns the OS lock."""
        return self._acquired

    def acquire(self) -> RunLock:
        """Acquire ownership or fail after the configured bounded timeout."""
        self.lock_root.mkdir(parents=True, exist_ok=True)
        try:
            self._lock.acquire()
        except Timeout as exc:
            raise RunLockedError(
                f"run {self.run_id!r} is already locked at {self.path}",
                hint=(
                    "wait for the owning process to finish or retry with a bounded "
                    "lock timeout; do not delete an active lock file"
                ),
            ) from exc
        self._acquired = True
        try:
            write_json_atomic(
                self.metadata_path,
                {
                    "pid": os.getpid(),
                    "started_at": utc_now(),
                },
            )
        except BaseException as exc:
            self._lock.release()
            self._acquired = False
            raise LifecycleError(
                f"could not publish run-lock diagnostics at {self.metadata_path}: {exc}"
            ) from exc
        return self

    def release(self) -> None:
        """Release the actual lock; stale diagnostic files are never ownership."""
        if not self._acquired:
            return
        try:
            with contextlib.suppress(OSError):
                self.metadata_path.unlink()
        finally:
            self._lock.release()
            self._acquired = False

    def __enter__(self) -> RunLock:
        return self.acquire()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any,
    ) -> None:
        self.release()
