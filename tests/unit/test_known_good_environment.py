from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_known_good_environment_is_machine_readable_and_consistent() -> None:
    from scripts.check_known_good_environment import check_known_good_environment

    assert check_known_good_environment(ROOT) == []


def test_known_good_environment_is_scoped_to_one_measured_stack() -> None:
    payload = json.loads(
        (ROOT / "environments/known-good-rtx4080-cu130.json").read_text(encoding="utf-8")
    )
    assert payload["schema_version"] == 1
    assert payload["status"] == "maintainer_measured"
    assert payload["required"]["gpu_name"] == "NVIDIA GeForce RTX 4080"
    assert payload["scope"]["other_hardware"] == "unmeasured"
    assert payload["pytorch"]["index_url"].startswith("https://download.pytorch.org/whl/")
    assert "+cu" in payload["required"]["packages"]["torch"]
    assert payload["observed"]["driver"] == "596.49"
