"""Version-bound Hydra composition of data-only config trees and exact argv."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import posixpath
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml

from miniverl.algorithms.contract import UPSTREAM_VERL_COMMIT, UPSTREAM_VERL_TAG
from miniverl.bridge.direct import _digest
from miniverl.errors import ConfigError


class _UniqueLoader(yaml.SafeLoader):
    def construct_mapping(self, node: Any, deep: bool = False) -> Any:
        keys = [self.construct_object(key, deep=deep) for key, _ in node.value]
        if len(set(keys)) != len(keys):
            raise ConfigError("duplicate Hydra YAML mapping key")
        return super().construct_mapping(node, deep=deep)


@dataclass(frozen=True)
class ComposedVerl:
    resolved_yaml: bytes
    provenance: dict[str, Any]


def packaged_tree() -> dict[str, Any]:
    return json.loads(files("miniverl.resources").joinpath("verl_v09_hydra_tree.json").read_text())


def _screen(value: Any) -> None:
    from miniverl.utils.privacy import _CREDENTIAL_VALUE, _is_sensitive_key

    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ConfigError("Hydra YAML requires string mapping keys")
            if key == "hydra" or key == "searchpath":
                raise ConfigError(
                    "Hydra runtime/search-path configuration is not a training override"
                )
            if (
                _is_sensitive_key(key)
                and item is not None
                and not (key == "continuous_token" and isinstance(item, dict))
                and not (
                    key
                    in {
                        "balance_dp_token",
                        "continuous_token",
                        "forward_max_token_len_per_gpu",
                        "log_prob_max_token_len_per_gpu",
                        "max_token_len_per_gpu",
                        "ppo_max_token_len_per_gpu",
                    }
                    and (
                        isinstance(item, (int, float, bool))
                        or (isinstance(item, str) and item.startswith("${"))
                    )
                )
            ):
                # Known model fields such as max_tokens are not credential keys.
                raise ConfigError("credentials belong in the process environment, not Hydra config")
            _screen(item)
    elif isinstance(value, list):
        for item in value:
            _screen(item)
    elif isinstance(value, str):
        if value.lower() in {"nan", ".nan", "inf", ".inf", "+inf", "-inf", "infinity"}:
            raise ConfigError("Hydra values must be finite")
        if _CREDENTIAL_VALUE.search(value):
            raise ConfigError("credentials belong in the process environment, not Hydra overrides")
        for resolver in re.findall(r"\$\{\s*([\w.]+):", value):
            if resolver != "oc.select":
                raise ConfigError(
                    f"unsupported Hydra resolver {resolver}; use references or oc.select"
                )


def _yaml(content: bytes, *, name: str) -> dict[str, Any]:
    try:
        value = yaml.load(content, Loader=_UniqueLoader)
        if not isinstance(value, dict):
            raise ConfigError("Hydra config file must contain a mapping")
        _screen(value)
        _digest(value)
        defaults = value.get("defaults", [])
        references = []
        for entry in defaults:
            references.extend([*entry, *entry.values()] if isinstance(entry, dict) else [entry])
        for reference in references:
            if not isinstance(reference, str):
                continue
            reference = reference.removeprefix("override ").removeprefix("optional ").split("@")[0]
            normalized = posixpath.normpath(posixpath.join(posixpath.dirname(name), reference))
            if "://" in reference or normalized == ".." or normalized.startswith("../"):
                raise ConfigError("Hydra defaults must stay within the pinned config tree")
        return value
    except (ValueError, TypeError, yaml.YAMLError, RecursionError) as exc:
        raise ConfigError(f"invalid data-only Hydra YAML: {exc}") from exc


def compose_verl(
    *,
    config_name: str,
    overrides: list[str],
    config_path: Path | str | None = None,
) -> ComposedVerl:
    """Snapshot, validate and compose without importing upstream application code.

    Paths may contain added local YAML configs, but the upstream base files must
    match the packaged pin. CRLF/LF conversion is identity-neutral; exact input
    file hashes are also retained. Composition only sees the captured snapshot.
    """
    if (
        not re.fullmatch(r"[\w/-]+(?:\.yaml)?", config_name)
        or ".." in config_name
        or config_name.startswith("/")
    ):
        raise ConfigError("config name must be a relative name inside the pinned Hydra tree")
    try:
        from hydra.core.override_parser.overrides_parser import OverridesParser
    except ImportError as exc:
        raise ConfigError("native Hydra ingestion requires pip install 'miniverl[hydra]'") from exc
    if importlib.metadata.version("hydra-core") != "1.3.2":
        raise ConfigError(
            "this config-tree adapter requires hydra-core==1.3.2; install miniverl[hydra]"
        )
    for item in overrides:
        _screen(item)
        if "../" in item or "..\\" in item or "pkg://" in item or "file://" in item:
            raise ConfigError("Hydra overrides cannot escape the captured config tree")
        try:
            parsed = OverridesParser.create().parse_override(item)
            if parsed.is_sweep_override():
                raise ConfigError("Hydra sweeps are not one training run; supply a single value")
            key = parsed.key_or_group
            _screen({part: parsed.value() for part in key.split(".")})
            if key.startswith("hydra") or ".." in key or "://" in key:
                raise ConfigError("Hydra search path/runtime overrides are not supported")
        except ConfigError:
            raise
        except Exception as exc:
            raise ConfigError("invalid Hydra override; use a single key=value assignment") from exc
    pin = packaged_tree()
    if pin["upstream_commit"] != UPSTREAM_VERL_COMMIT:
        raise ConfigError("packaged Hydra tree does not match the supported upstream version")
    captured = {}
    if config_path is None:
        captured = {name: record["text"].encode() for name, record in pin["files"].items()}
    else:
        root = Path(config_path).resolve()
        if not root.is_dir():
            raise ConfigError(
                "--verl-config-path must name the pinned verl/trainer/config directory"
            )
        for path in sorted(root.rglob("*.yaml")):
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                raise ConfigError("Hydra config paths cannot escape through symlinks")
            if path.stat().st_size > 4 * 1024 * 1024:
                raise ConfigError("Hydra config file exceeds 4 MiB")
            captured[path.relative_to(root).as_posix()] = path.read_bytes()
            if len(captured) > 256 or sum(map(len, captured.values())) > 8 * 1024 * 1024:
                raise ConfigError("Hydra config tree exceeds the 256-file / 8 MiB input bound")
        for name, record in pin["files"].items():
            content = captured.get(name, b"").replace(b"\r\n", b"\n")
            if hashlib.sha256(content).hexdigest() != record["sha256"]:
                raise ConfigError(
                    f"config tree differs from pinned verl {UPSTREAM_VERL_TAG}: {name}; "
                    "use that release's config tree and express local changes as overrides or an added config"
                )
    if len(captured) > 256 or sum(map(len, captured.values())) > 8 * 1024 * 1024:
        raise ConfigError("Hydra config tree exceeds the 256-file / 8 MiB input bound")
    defaults = {}
    for name, content in captured.items():
        value = _yaml(content, name=name)
        if "defaults" in value:
            defaults[name] = value["defaults"]
    with tempfile.TemporaryDirectory(prefix="miniverl-hydra-") as temporary:
        root = Path(temporary)
        for name, content in captured.items():
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        request = {"directory": str(root), "name": config_name, "overrides": overrides}
        try:
            result = subprocess.run(
                [sys.executable, "-I", str(Path(__file__).with_name("hydra_worker.py"))],
                input=json.dumps(request),
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=60,
                check=False,
            )
            response = json.loads(result.stdout)
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            raise ConfigError("isolated Hydra composition failed or timed out") from exc
        if result.returncode or "error" in response:
            raise ConfigError(
                f"Hydra composition failed: {str(response.get('error', 'unknown'))[:2000]}"
            )
    value = response["value"]
    _screen(value)
    try:
        _digest(value)
    except (ValueError, TypeError) as exc:
        raise ConfigError(
            "resolved Hydra config must contain finite JSON-compatible values"
        ) from exc
    if "${" in json.dumps(value):
        raise ConfigError("Hydra left an unresolved interpolation; supply its referenced value")
    rendered = yaml.safe_dump(value, sort_keys=True, allow_unicode=True).encode()
    hashes = {name: hashlib.sha256(content).hexdigest() for name, content in captured.items()}
    return ComposedVerl(
        rendered,
        {
            "composition_status": "resolved",
            "adapter": "verl-v0.9-hydra-v1",
            "upstream_tag": UPSTREAM_VERL_TAG,
            "upstream_commit": UPSTREAM_VERL_COMMIT,
            "hydra_version": "1.3.2",
            "config_path": str(config_path) if config_path else "packaged",
            "config_name": config_name,
            "config_defaults": defaults,
            "overrides": list(overrides),
            "input_file_sha256": hashes,
            "input_tree_sha256": _digest(hashes),
            "resolved_sha256": hashlib.sha256(rendered).hexdigest(),
        },
    )
