"""Domain-specific exceptions.

Every miniVERL error carries a human-readable ``message`` and an optional
``hint`` describing the concrete next action.  The CLI renders ``hint`` on a
separate line, so hints should be imperative and copy-pasteable.
"""

from __future__ import annotations

__all__ = [
    "MiniVerlError",
    "ConfigError",
    "SchemaValidationError",
    "TrajectoryError",
    "AlignmentError",
    "TokenizerMismatchError",
    "ToolCallParseError",
    "ToolEnvironmentError",
    "CacheError",
    "StaleCacheError",
    "CacheCorruptionError",
    "MissingDependencyError",
    "BackendError",
    "MemoryStrategyError",
    "GpuMemoryError",
    "CheckpointError",
    "ReportError",
    "RunNotFoundError",
]


class MiniVerlError(Exception):
    """Base class for every error raised on purpose by miniVERL."""

    def __init__(self, message: str, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    def __str__(self) -> str:
        if self.hint:
            return f"{self.message}\n  hint: {self.hint}"
        return self.message


class ConfigError(MiniVerlError):
    """A run configuration is malformed, contradictory or unsupported."""


class SchemaValidationError(MiniVerlError):
    """A serialized artifact does not match its declared schema."""


class TrajectoryError(MiniVerlError):
    """A trajectory violates the span/mask/token invariants."""


class AlignmentError(MiniVerlError):
    """Student and teacher prediction positions could not be aligned."""


class TokenizerMismatchError(AlignmentError):
    """Student and teacher tokenizers differ; miniVERL requires an identical one."""


class ToolCallParseError(MiniVerlError):
    """The model emitted text that is not a valid tool call or final answer."""


class ToolEnvironmentError(MiniVerlError):
    """A tool environment rejected an operation or failed to execute one."""


class CacheError(MiniVerlError):
    """Generic teacher-target cache failure."""


class StaleCacheError(CacheError):
    """A cache entry was produced by an incompatible policy/model/config."""


class CacheCorruptionError(CacheError):
    """A cache shard failed its checksum or structural validation."""


class MissingDependencyError(MiniVerlError):
    """An optional dependency is required for the requested operation."""

    def __init__(self, package: str, extra: str, purpose: str) -> None:
        message = f"{purpose} requires the optional dependency '{package}', which is not installed."
        hint = f'pip install "miniverl[{extra}]"'
        super().__init__(message, hint)
        self.package = package
        self.extra = extra


class BackendError(MiniVerlError):
    """A model backend could not be constructed or used as configured."""


class MemoryStrategyError(MiniVerlError):
    """A memory strategy could not honour its contract."""


class GpuMemoryError(MiniVerlError):
    """CUDA ran out of memory and the bounded, equivalence-preserving retries failed."""


class CheckpointError(MiniVerlError):
    """A checkpoint could not be written, read or resumed."""


class ReportError(MiniVerlError):
    """A report could not be produced from the run artifacts."""


class RunNotFoundError(MiniVerlError):
    """A run directory does not exist or is missing required artifacts."""
