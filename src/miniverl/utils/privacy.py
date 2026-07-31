"""Canonical privacy-safe views for reports and portable artifacts."""

from __future__ import annotations

import html
import re
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any

import yaml

__all__ = ["portable_payload", "portable_text", "portable_yaml"]

_KEY_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|[^A-Za-z0-9]+")
_SECRET_SUFFIXES = {"token", "secret", "password", "credential"}
_SECRET_KEY_PAIRS = {("api", "key"), ("access", "key"), ("private", "key")}
_SENSITIVE_WHOLE_KEYS = {"authorization", "cookie", "session"}
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
    r"(?i)\b("
    r"(?:[A-Za-z][A-Za-z0-9_-]*)?(?:token|secret|password|credential|api[_-]?key)"
    r"|authorization|cookie|session"
    r")\s*[:=]\s*(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)"
)
_BEARER_SECRET = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}")
_URL_USERINFO = re.compile(r"(?i)\b(https?://)[^/@\s]+@")
_PATH_END = r"(?=$|[\r\n\t\"'<>|;,)]|\s+(?:https?://|[A-Za-z_][\w-]*\s*[:=]))"
_WINDOWS_PATH = re.compile(
    rf"(?i)(?<![A-Za-z0-9])"
    rf"(?:[A-Z]:[\\/]|(?:\\\\|//)[^\\/\s]+[\\/][^\\/\s]+[\\/]).+?{_PATH_END}"
)
_POSIX_PRIVATE_PATH = re.compile(
    rf"(?<![A-Za-z0-9])/(?:Users|home|tmp|private|var/tmp)/.+?{_PATH_END}"
)
_ENV_REFERENCE = re.compile(r"^(?:%[A-Za-z_][A-Za-z0-9_]*%|\$\{?[A-Za-z_][A-Za-z0-9_]*\}?)$")


def _portable_path(value: str) -> str | None:
    windows = PureWindowsPath(value)
    posix = PurePosixPath(value)
    if windows.is_absolute() or posix.is_absolute():
        name = windows.name if windows.is_absolute() else posix.name
        return f"<local>/{name or 'path'}"
    return None


def _is_sensitive_key(key: str) -> bool:
    """Recognize semantic secret suffixes without matching ``tokenizer_id``."""
    components = [component.lower() for component in _KEY_BOUNDARY.split(key) if component]
    if not components:
        return False
    if len(components) == 1 and components[0] in _SENSITIVE_WHOLE_KEYS:
        return True
    if components[-1] in _SECRET_SUFFIXES:
        return True
    return len(components) >= 2 and tuple(components[-2:]) in _SECRET_KEY_PAIRS


def _replace_path(match: re.Match[str]) -> str:
    path = match.group(0).rstrip()
    trailing = match.group(0)[len(path) :]
    return (_portable_path(path) or "<local>/path") + trailing


def portable_text(value: str) -> str:
    """Remove local paths, credential-like values, and environment references."""
    value = html.unescape(value)
    exact_path = _portable_path(value)
    if exact_path is not None:
        return exact_path
    if _ENV_REFERENCE.fullmatch(value.strip()):
        return "<environment-variable>"
    redacted = _CREDENTIAL_VALUE.sub("<redacted>", value)
    redacted = _BEARER_SECRET.sub("Bearer <redacted>", redacted)
    redacted = _ASSIGNED_SECRET.sub(lambda match: f"{match.group(1)}=<redacted>", redacted)
    redacted = _URL_USERINFO.sub(r"\1<redacted>@", redacted)
    redacted = _WINDOWS_PATH.sub(_replace_path, redacted)
    redacted = _POSIX_PRIVATE_PATH.sub(_replace_path, redacted)
    return redacted


def portable_payload(value: Any, *, key: str | None = None) -> Any:
    """Recursively construct the canonical portable/redacted artifact view."""
    normalized_key = (key or "").lower()
    if _is_sensitive_key(key or "") or normalized_key in _IDENTITY_KEYS:
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
