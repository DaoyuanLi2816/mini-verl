from __future__ import annotations

import json
from pathlib import Path

from scripts.publish_verl_opd_compatibility import render_official_report

ROOT = Path(__file__).resolve().parents[2]


def test_official_example_coverage_report_is_generated_and_self_consistent() -> None:
    target = ROOT / "docs/generated/verl-opd-v08-official-fields.json"
    assert target.read_text(encoding="utf-8") == render_official_report(ROOT)
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert sum(payload["classifications"].values()) == payload["official_example_fields_total"]
    assert payload["executable"] is False
    assert "algorithm.adv_estimator" in payload["unsupported_fields"]
