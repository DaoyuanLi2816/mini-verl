from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any


def _publisher() -> Any:
    path = Path("scripts/publish_verl_rl_v09_compatibility.py")
    spec = importlib.util.spec_from_file_location("publish_verl_rl_v09_compatibility", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_verl_v09_compatibility_report_is_exactly_generated() -> None:
    publisher = _publisher()
    committed = Path("docs/generated/verl-rl-v0.9-compatibility.json").read_text(encoding="utf-8")
    assert committed == publisher.render()
    payload = json.loads(committed)
    assert payload["status"] == "accepted"
    assert payload["executable"] is True
    assert payload["algorithm"] == "grpo"
    assert payload["source_verl"] == {
        "repository": "https://github.com/verl-project/verl",
        "tag": "v0.9.0",
        "commit": "483b8a009ba3a97563edee3a19887e4862b8094a",
    }
    assert not payload["unsupported_fields"]
    assert not payload["unresolved_fields"]
    assert not payload["required_user_input"]
    classifications = {row["classification"] for row in payload["field_classification"]}
    assert "distributed_only" in classifications
    assert "locally_lowered" in classifications
