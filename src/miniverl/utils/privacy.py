"""Canonical privacy-safe views for reports and portable artifacts."""

from __future__ import annotations

import re
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any

import yaml

__all__ = ["portable_payload", "portable_text", "portable_yaml"]

_SECRET_KEY = re.compile(
    r"^(?:token|api_?token|auth_?token|access_?token|refresh_?token|"
    r"secret|client_secret|password|credential|private_key|access_key|api_?key)$",
    re.IGNORECASE,
)
_IDENTITY_KEYS = {
    "hostname",
    "host_name",
    "computer_name",
    "username",
    "user_name",
    "home_directory",
}
_CREDENTIAL_VALUE = re.compile(
    r"\b(?:gh[oprsu]_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{20,}|pypi-[A-Za-z0-9_-]{20,})\b"
)
_ASSIGNED_SECRET = re.compile(
    r"(?i)\b(token|secret|password|credential|api[_-]?key)\s*[:=]\s*[^\s,;]+"
)
_BEARER_SECRET = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}")
_WINDOWS_PATH = re.compile(r"(?i)\b[A-Z]:[\\/][^\s\"'<>|]+")
_POSIX_PRIVATE_PATH = re.compile(
    r"(?<![A-Za-z0-9])/(?:Users|home|tmp|private|var/tmp)/[^\s\"'<>|]+"
)
_ENV_REFERENCE = re.compile(r"^(?:%[A-Za-z_][A-Za-z0-9_]*%|\$\{?[A-Za-z_][A-Za-z0-9_]*\}?)$")


def _portable_path(value: str) -> str | None:
    windows = PureWindowsPath(value)
    posix = PurePosixPath(value)
    if windows.is_absolute() or posix.is_absolute():
        name = windows.name if windows.is_absolute() else posix.name
        return f"<local>/{name or 'path'}"
    return None


def portable_text(value: str) -> str:
    """Remove local paths, credential-like values, and environment references."""
    exact_path = _portable_path(value)
    if exact_path is not None:
        return exact_path
    if _ENV_REFERENCE.fullmatch(value.strip()):
        return "<environment-variable>"
    redacted = _CREDENTIAL_VALUE.sub("<redacted>", value)
    redacted = _ASSIGNED_SECRET.sub(lambda match: f"{match.group(1)}=<redacted>", redacted)
    redacted = _BEARER_SECRET.sub("Bearer <redacted>", redacted)
    redacted = _WINDOWS_PATH.sub(
        lambda match: _portable_path(match.group(0)) or "<local>/path",
        redacted,
    )
    redacted = _POSIX_PRIVATE_PATH.sub(
        lambda match: _portable_path(match.group(0)) or "<local>/path",
        redacted,
    )
    return redacted


def portable_payload(value: Any, *, key: str | None = None) -> Any:
    """Recursively construct the canonical portable/redacted artifact view."""
    normalized_key = (key or "").lower()
    if _SECRET_KEY.search(normalized_key) or normalized_key in _IDENTITY_KEYS:
        return "<redacted>"
    if isinstance(value, dict):
        return {
            item_key: portable_payload(item, key=str(item_key)) for item_key, item in value.items()
        }
    if isinstance(value, list):
        return [portable_payload(item) for item in value]
    if isinstance(value, tuple):
        return [portable_payload(item) for item in value]
    if isinstance(value, str):
        return portable_text(value)
    return value


def portable_yaml(text: str) -> str:
    """Return a canonical sanitized YAML view without leaking parse failures."""
    try:
        payload = yaml.safe_load(text)
    except yaml.YAMLError:
        return portable_text(text)
    portable = portable_payload(payload)
    return yaml.safe_dump(portable, sort_keys=False, allow_unicode=True, width=100)
