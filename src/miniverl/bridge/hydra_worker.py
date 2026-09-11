"""Isolated composition adapter. Never instantiate targets or import verl."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch


def compose_request(request: dict[str, Any]) -> dict[str, Any]:
    from hydra import compose, initialize_config_dir
    from hydra._internal.core_plugins.file_config_source import FileConfigSource
    from omegaconf import OmegaConf

    # The Python installation is trusted, just as the installed Hydra package
    # is. This adapter is not an OS sandbox. Configs cannot add plugin search
    # paths, and interpolated defaults cannot load files outside the snapshot.
    root = Path(request["directory"]).resolve()
    original = FileConfigSource.load_config

    def confined_load(source: Any, config_path: str) -> Any:
        target = (Path(source.path) / source._normalize_file_name(config_path)).resolve()
        if not target.is_relative_to(root):
            raise ValueError("Hydra defaults escaped the captured config tree")
        return original(source, config_path)

    with (
        patch.object(FileConfigSource, "load_config", confined_load),
        initialize_config_dir(version_base="1.3", config_dir=str(root)),
    ):
        config = compose(config_name=request["name"], overrides=request["overrides"])
        value = OmegaConf.to_container(config, resolve=True, throw_on_missing=False)
    return {"value": value}


def main() -> None:
    try:
        result = compose_request(json.load(sys.stdin))
        print(json.dumps(result, allow_nan=False))
    except Exception as exc:
        # Error text is bounded by the parent, which has already screened
        # credentials and denied environment/custom resolvers.
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
