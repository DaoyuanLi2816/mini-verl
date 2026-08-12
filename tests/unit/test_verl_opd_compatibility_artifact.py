from __future__ import annotations

import json
from pathlib import Path


def test_published_opd_compatibility_matrix_is_compiler_bound() -> None:
    from miniverl.utils.runs import canonical_json
    from scripts.publish_verl_opd_compatibility import build_matrix

    source = Path("examples/verl-opd-v0.8-single-gpu.yaml")
    artifact = Path("docs/generated/verl-opd-v0.8-compatibility.json")
    expected = canonical_json(build_matrix(source))

    assert artifact.read_text(encoding="utf-8") == expected
    payload = json.loads(expected)
    assert payload["profile"] == "verl-opd-v0.8-single-gpu-v1"
    assert payload["field_count"] >= 70
    assert payload["executable"] is True
    assert payload["upstream"]["commit"] == "7aed6b230776f963fa09509c10d9c3a767d1102c"
    assert "distributed execution" in payload["scope"]
