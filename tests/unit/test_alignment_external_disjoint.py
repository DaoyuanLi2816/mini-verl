"""A pre-final-test decision cannot be made on a final-test task.

The starting checkpoint and the teacher are both chosen before the final test is
read. If either selection suite drew a task the final test also scores, the
"one read" of the final test would already have leaked into the choices the
study claims were made without it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from miniverl.alignment_external.suite import prepare_suite

PROFILE = {
    "id": "selection",
    "selection_seed": 11,
    "endpoints": [{"id": "ifeval", "tasks": 16}],
}


def _resolver(count: int) -> Any:
    def resolve(endpoint: dict[str, Any]) -> tuple[list[str], list[str] | None]:
        return [f"ifeval-{index:04d}" for index in range(count)], None

    return resolve


def test_reserved_tasks_never_appear_in_the_selection(tmp_path: Path) -> None:
    reserved = [f"ifeval-{index:04d}" for index in range(40)]

    manifest = prepare_suite(
        profile=PROFILE,
        out=tmp_path,
        resolver=_resolver(120),
        reserved_task_ids={"ifeval": reserved},
    )

    entry = manifest["endpoints"][0]
    assert entry["reserved_for_final_test"] == 40
    assert not set(entry["task_ids"]) & set(reserved)
    assert len(entry["task_ids"]) == 16


def test_an_exhausted_pool_fails_closed(tmp_path: Path) -> None:
    """Better to stop than to quietly select fewer tasks than requested."""
    reserved = [f"ifeval-{index:04d}" for index in range(110)]

    with pytest.raises(ValueError, match="reserved for the final test"):
        prepare_suite(
            profile=PROFILE,
            out=tmp_path,
            resolver=_resolver(120),
            reserved_task_ids={"ifeval": reserved},
        )


def test_without_reservation_nothing_is_withheld(tmp_path: Path) -> None:
    manifest = prepare_suite(profile=PROFILE, out=tmp_path, resolver=_resolver(120))

    assert manifest["endpoints"][0]["reserved_for_final_test"] == 0


def test_a_selection_suite_is_disjoint_from_the_frozen_final_suite(tmp_path: Path) -> None:
    """The real flow: freeze the final suite, then select against its manifest."""
    final_profile = {
        "id": "final",
        "selection_seed": 7,
        "endpoints": [{"id": "ifeval", "tasks": 32}],
    }
    final = prepare_suite(profile=final_profile, out=tmp_path / "final", resolver=_resolver(200))
    final_ids = final["endpoints"][0]["task_ids"]

    selection = prepare_suite(
        profile=PROFILE,
        out=tmp_path / "selection",
        resolver=_resolver(200),
        reserved_task_ids={"ifeval": final_ids},
    )
    selection_ids = selection["endpoints"][0]["task_ids"]

    assert set(final_ids) & set(selection_ids) == set()
    assert len(final_ids) == 32
    assert len(selection_ids) == 16
    # And the reservation is recorded, so the disjointness is auditable from
    # the manifest rather than only from having run this test.
    assert selection["endpoints"][0]["reserved_for_final_test"] == 32


def test_the_manifest_records_the_reservation_on_disk(tmp_path: Path) -> None:
    prepare_suite(
        profile=PROFILE,
        out=tmp_path,
        resolver=_resolver(120),
        reserved_task_ids={"ifeval": [f"ifeval-{index:04d}" for index in range(20)]},
    )

    written = json.loads((tmp_path / "suite-manifest.json").read_text(encoding="utf-8"))

    assert written["endpoints"][0]["reserved_for_final_test"] == 20
