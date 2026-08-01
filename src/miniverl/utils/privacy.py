"""Canonical privacy-safe views for reports and portable artifacts."""

from __future__ import annotations

import html
import re
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import yaml

__all__ = ["portable_payload", "portable_text", "portable_yaml"]

_KEY_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|[^A-Za-z0-9]+")
_SECRET_COMPONENTS = {"token", "secret", "password", "credential"}
_SECRET_KEY_PAIRS = {
    ("api", "key"),
    ("access", "key"),
    ("access", "token"),
    ("private", "key"),
    ("client", "secret"),
}
_BENIGN_KEY_COMPONENTS = {
    ("tokenizer",),
    ("tokenizer", "id"),
    ("tokenizer", "revision"),
    ("tokenizer", "fingerprint"),
    ("tokenizer", "fingerprints"),
    ("tokenizer", "identity"),
    ("token", "count"),
    ("token", "budget"),
    ("token", "type"),
    ("token", "id"),
    ("token", "ids"),
    ("token", "piece"),
    ("token", "loss"),
    ("token", "weights"),
    ("token", "analysis"),
    ("tokens",),
    ("tokens", "by", "span", "type"),
    ("selected", "tokens"),
    ("selected", "critical", "tokens"),
    ("selected", "model", "tokens"),
    ("selected", "training", "tokens", "total"),
    ("total", "critical", "tokens"),
    ("total", "model", "tokens"),
    ("generated", "tokens"),
    ("generated", "token", "count"),
    ("generated", "tokens", "per", "task"),
    ("generated", "training", "tokens", "total"),
    ("model", "generated", "training", "tokens", "total"),
    ("context", "tokens"),
    ("critical", "tokens"),
    ("model", "tokens"),
    ("max", "tokens"),
    ("max", "new", "tokens"),
    ("max", "new", "tokens", "per", "turn"),
    ("max", "tokens", "per", "trajectory"),
    ("max", "total", "tokens"),
    ("target", "token", "ids"),
    ("prefix", "token", "ids"),
    ("bos", "token", "id"),
    ("eos", "token", "id"),
    ("teacher", "token"),
    ("teacher", "top", "token"),
    ("student", "top", "token"),
    ("per", "token"),
    ("per", "token", "ce"),
    ("per", "token", "divergence"),
    ("per", "token", "objective"),
    ("sampled", "token", "nll", "weight"),
    ("model", "token", "mask"),
    ("protocol", "token", "accuracy"),
    ("tokens", "per", "solved", "task"),
    ("is", "pretokenized"),
    ("session", "length"),
    ("session", "count"),
    ("cookie", "count"),
}
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
_ASSIGNED_FIELD = re.compile(
    r"(?i)(?<![A-Za-z0-9_./-])(?P<key>[A-Za-z][A-Za-z0-9_.-]{0,100})"
    r"(?P<separator>\s*[:=]\s*)"
    r"(?P<value>\"[^\"]*\"|'[^']*'|[^\s,;]+)"
)
_BEARER_SECRET = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}")
_SUPPORTED_URL = re.compile(
    r"(?i)\b(?:https?|ssh|git\+ssh|postgresql|postgres|mysql|mongodb|redis)://[^\s<>\"']+"
)
_PATH_END = r"(?=$|[\r\n\t\"'<>|;,)]|\s+(?:https?://|[A-Za-z_][\w-]*\s*[:=]))"
_WINDOWS_PATH = re.compile(
    rf"(?i)(?<![A-Za-z0-9])"
    rf"(?:[A-Z]:[\\/]|(?:\\\\|//)[^\\/\s]+[\\/][^\\/\s]+[\\/]).+?{_PATH_END}"
)
_POSIX_ABSOLUTE_PATH = re.compile(rf"(?<![A-Za-z0-9:/])/(?:[^/\s<>\"']+)/.+?{_PATH_END}")
_ENV_REFERENCE = re.compile(r"^(?:%[A-Za-z_][A-Za-z0-9_]*%|\$\{?[A-Za-z_][A-Za-z0-9_]*\}?)$")


def _portable_path(value: str) -> str | None:
    windows = PureWindowsPath(value)
    posix = PurePosixPath(value)
    if windows.is_absolute() or posix.is_absolute():
        name = windows.name if windows.is_absolute() else posix.name
        return f"<local>/{name or 'path'}"
    return None


def _is_sensitive_key(key: str) -> bool:
    """Recognize semantic credential components with a narrow metadata allowlist."""
    components = tuple(component.lower() for component in _KEY_BOUNDARY.split(key) if component)
    if not components:
        return False
    if components in _BENIGN_KEY_COMPONENTS:
        return False
    component_set = set(components)
    if "authorization" in component_set or "cookie" in component_set:
        return True
    if "session" in component_set and (
        len(components) == 1
        or component_set.intersection({"id", "key", "token", "secret", "cookie"})
    ):
        return True
    if component_set.intersection(_SECRET_COMPONENTS):
        return True
    return any(set(pair).issubset(component_set) for pair in _SECRET_KEY_PAIRS)


def _replace_path(match: re.Match[str]) -> str:
    path = match.group(0).rstrip()
    trailing = match.group(0)[len(path) :]
    return (_portable_path(path) or "<local>/path") + trailing


def _replace_assigned_field(match: re.Match[str]) -> str:
    if not _is_sensitive_key(match.group("key")):
        return match.group(0)
    return f"{match.group('key')}=<redacted>"


def _sanitize_url(match: re.Match[str]) -> str:
    raw = match.group(0)
    trailing = ""
    while raw and raw[-1] in ".,;)]}":
        trailing = raw[-1] + trailing
        raw = raw[:-1]
    try:
        parsed = urlsplit(raw)
        if parsed.username is None and parsed.password is None:
            return raw + trailing
        hostname = parsed.hostname
        if not hostname:
            return f"{parsed.scheme}://<redacted>@" + trailing
        host = f"[{hostname}]" if ":" in hostname and not hostname.startswith("[") else hostname
        port = f":{parsed.port}" if parsed.port is not None else ""
        sanitized = urlunsplit(
            (parsed.scheme, f"<redacted>@{host}{port}", parsed.path, parsed.query, parsed.fragment)
        )
        return sanitized + trailing
    except ValueError:
        scheme, separator, remainder = raw.partition("://")
        _userinfo, at, host_and_path = remainder.rpartition("@")
        return (f"{scheme}{separator}<redacted>@{host_and_path}" if at else raw) + trailing


def portable_text(value: str) -> str:
    """Remove local paths, credential-like values, and environment references."""
    value = html.unescape(value)
    exact_path = _portable_path(value)
    if exact_path is not None:
        return exact_path
    if _ENV_REFERENCE.fullmatch(value.strip()):
        return "<environment-variable>"
    protected_urls: list[str] = []

    def protect_url(match: re.Match[str]) -> str:
        protected_urls.append(_sanitize_url(match))
        return f"MINIVERLPROTECTEDURL{len(protected_urls) - 1}END"

    redacted = _SUPPORTED_URL.sub(protect_url, value)
    redacted = _CREDENTIAL_VALUE.sub("<redacted>", redacted)
    redacted = _BEARER_SECRET.sub("Bearer <redacted>", redacted)
    redacted = _ASSIGNED_FIELD.sub(_replace_assigned_field, redacted)
    redacted = _WINDOWS_PATH.sub(_replace_path, redacted)
    redacted = _POSIX_ABSOLUTE_PATH.sub(_replace_path, redacted)
    for index, sanitized_url in enumerate(protected_urls):
        redacted = redacted.replace(f"MINIVERLPROTECTEDURL{index}END", sanitized_url)
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
