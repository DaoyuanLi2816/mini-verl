"""Structured logging.

Two sinks, one call site: a human-readable Rich line on the console and a
machine-readable JSON object appended to ``events.jsonl``.  Nothing is sent
anywhere else -- miniVERL has no telemetry.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from rich.console import Console
from rich.logging import RichHandler

__all__ = ["get_console", "configure_logging", "get_logger", "EventLog"]

_CONSOLE: Console | None = None


def get_console(*, stderr: bool = False) -> Console:
    """Shared Rich console."""
    global _CONSOLE
    if _CONSOLE is None or _CONSOLE.stderr != stderr:
        _CONSOLE = Console(stderr=stderr, soft_wrap=False)
    return _CONSOLE


def configure_logging(level: str | None = None) -> None:
    """Install a Rich log handler once, honouring ``MINIVERL_LOG_LEVEL``.

    An explicit ``level`` always wins.  Passing ``None`` after logging is already
    configured leaves the level alone, so an earlier ``--log-level DEBUG`` is not
    silently downgraded by the first ``get_logger()`` call a module makes.
    """
    root = logging.getLogger("miniverl")
    if root.handlers:
        if level is not None:
            root.setLevel(level.upper())
        return
    resolved = (level or os.environ.get("MINIVERL_LOG_LEVEL") or "INFO").upper()
    handler = RichHandler(
        console=get_console(stderr=True),
        show_path=False,
        show_time=False,
        rich_tracebacks=False,
        markup=False,
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(handler)
    root.setLevel(resolved)
    root.propagate = False


def get_logger(name: str = "miniverl") -> logging.Logger:
    """Return a namespaced logger."""
    configure_logging()
    return logging.getLogger(name if name.startswith("miniverl") else f"miniverl.{name}")


class EventLog:
    """Writes lifecycle events to ``events.jsonl`` and mirrors them to the log."""

    def __init__(self, writer: Any | None = None, *, logger_name: str = "miniverl.run") -> None:
        self.writer = writer
        self.log = get_logger(logger_name)

    def emit(self, event: str, /, level: int = logging.INFO, **fields: Any) -> dict[str, Any]:
        """Record one event and return the payload."""
        from miniverl.utils.runs import utc_now

        payload: dict[str, Any] = {"ts": utc_now(), "event": event, **fields}
        if self.writer is not None:
            self.writer.write(payload)
        detail = " ".join(f"{k}={v}" for k, v in fields.items() if not isinstance(v, (dict, list)))
        self.log.log(level, f"{event} {detail}".strip())
        return payload
