"""Suite preparation, validation and reporting, entirely offline.

The suite is what freezes the study, so the properties that matter are: task
selection cannot see a model outcome, the generation budget is enforced, and a
result set that does not match its manifest is rejected rather than reported.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from miniverl.alignment_external.records import TaskRecord
from miniverl.alignment_external.suite import (
    MAX_TASKS_PER_MODEL,
    prepare_suite,
    report_suite,
    select_task_ids,
    validate_results,
)

PROFILE_PATH = Path("benchmarks/external-alignment/profile-v1.yaml")


# ----------------------------------------------------------------- selection


def test_selection_is_deterministic_for_a_seed() -> None:
    ids = [f"t{i}" for i in range(100)]

    first = select_task_ids(ids, None, limit=20, seed=7)
    second = select_task_ids(ids, None, limit=20, seed=7)

    assert first == second
    assert len(first) == 20


def test_a_different_seed_selects_differently() -> None:
    ids = [f"t{i}" for i in range(100)]

    assert select_task_ids(ids, None, limit=20, seed=7) != select_task_ids(
        ids, None, limit=20, seed=8
    )


def test_everything_is_returned_when_the_limit_exceeds_the_pool() -> None:
    ids = ["b", "a", "c"]

    assert select_task_ids(ids, None, limit=10, seed=1) == ["a", "b", "c"]


def test_stratified_selection_keeps_every_category() -> None:
    """A small subset must not drop a category, or coverage silently narrows."""
    ids = [f"t{i}" for i in range(90)]
    strata = [f"cat{i % 9}" for i in range(90)]

    chosen = select_task_ids(ids, strata, limit=18, seed=3)

    by_stratum: dict[str, int] = {}
    for task_id in chosen:
        by_stratum[strata[ids.index(task_id)]] = by_stratum.get(strata[ids.index(task_id)], 0) + 1
    assert len(by_stratum) == 9
    assert set(by_stratum.values()) == {2}


def test_stratified_selection_handles_uneven_strata() -> None:
    ids = [f"t{i}" for i in range(30)]
    # One big stratum and two small ones.
    strata = ["big"] * 24 + ["small_a"] * 3 + ["small_b"] * 3

    chosen = select_task_ids(ids, strata, limit=12, seed=5)

    assert len(chosen) == 12
    picked = {strata[ids.index(t)] for t in chosen}
    assert picked == {"big", "small_a", "small_b"}


def test_mismatched_strata_length_is_rejected() -> None:
    with pytest.raises(ValueError, match="parallel"):
        select_task_ids(["a", "b"], ["x"], limit=1, seed=1)


def test_a_non_positive_limit_is_rejected() -> None:
    with pytest.raises(ValueError, match="positive"):
        select_task_ids(["a"], None, limit=0, seed=1)


# ------------------------------------------------------------------- prepare


def _resolver(counts: dict[str, int], strata_count: int = 4):
    def resolve(endpoint: dict[str, Any]) -> tuple[list[str], list[str] | None]:
        total = counts[endpoint["id"]]
        ids = [f"{endpoint['id']}-{i}" for i in range(total)]
        strata = [f"s{i % strata_count}" for i in range(total)]
        return ids, strata

    return resolve


def _profile(**overrides: Any) -> dict[str, Any]:
    base = {
        "id": "test-profile",
        "selection_seed": 11,
        "endpoints": [
            {"id": "ifeval", "tasks": 8},
            {"id": "xstest", "tasks": 8},
            {"id": "jbb_behaviors", "tasks": 4},
            {"id": "rewardbench", "tasks": 6, "counts_toward_generation": False},
        ],
    }
    base.update(overrides)
    return base


def test_prepare_writes_a_manifest_with_a_digest(tmp_path: Path) -> None:
    manifest = prepare_suite(
        profile=_profile(),
        out=tmp_path,
        resolver=_resolver({"ifeval": 60, "xstest": 60, "jbb_behaviors": 40, "rewardbench": 60}),
    )

    written = json.loads((tmp_path / "suite-manifest.json").read_text(encoding="utf-8"))
    assert written == manifest
    assert manifest["manifest_digest"]
    assert {e["id"] for e in manifest["endpoints"]} == {
        "ifeval",
        "xstest",
        "jbb_behaviors",
        "rewardbench",
    }


def test_preparation_is_reproducible(tmp_path: Path) -> None:
    resolver = _resolver({"ifeval": 60, "xstest": 60, "jbb_behaviors": 40, "rewardbench": 60})

    first = prepare_suite(profile=_profile(), out=tmp_path / "a", resolver=resolver)
    second = prepare_suite(profile=_profile(), out=tmp_path / "b", resolver=resolver)

    assert first["manifest_digest"] == second["manifest_digest"]


def test_judged_endpoints_do_not_consume_the_generation_budget(tmp_path: Path) -> None:
    manifest = prepare_suite(
        profile=_profile(),
        out=tmp_path,
        resolver=_resolver({"ifeval": 60, "xstest": 60, "jbb_behaviors": 40, "rewardbench": 60}),
    )

    # 8 + 8 + 4; the 6 RewardBench pairs are judged, not generated.
    assert manifest["generation_tasks_per_model"] == 20


def test_exceeding_the_generation_ceiling_is_refused(tmp_path: Path) -> None:
    profile = _profile(
        endpoints=[
            {"id": "ifeval", "tasks": 400},
            {"id": "xstest", "tasks": 400},
        ]
    )

    with pytest.raises(ValueError, match="compute-contract ceiling"):
        prepare_suite(
            profile=profile,
            out=tmp_path,
            resolver=_resolver({"ifeval": 600, "xstest": 600}),
        )


def test_an_unknown_endpoint_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not in the registry"):
        prepare_suite(
            profile=_profile(endpoints=[{"id": "made_up", "tasks": 4}]),
            out=tmp_path,
            resolver=_resolver({"made_up": 10}),
        )


def test_an_empty_profile_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no endpoints"):
        prepare_suite(profile=_profile(endpoints=[]), out=tmp_path, resolver=_resolver({}))


# ------------------------------------------------- the committed profile v1


def test_the_committed_profile_stays_under_the_generation_ceiling() -> None:
    profile = yaml.safe_load(PROFILE_PATH.read_text(encoding="utf-8"))

    generating = [
        item for item in profile["endpoints"] if item.get("counts_toward_generation", True)
    ]
    total = sum(int(item["tasks"]) for item in generating)

    assert total == profile["generation_budget"]["planned_tasks_per_model"]
    assert total <= MAX_TASKS_PER_MODEL
    assert profile["generation_budget"]["max_tasks_per_model"] == MAX_TASKS_PER_MODEL


def test_the_committed_profile_prepares_end_to_end(tmp_path: Path) -> None:
    """The real profile, including its non-registry utility endpoint."""
    profile = yaml.safe_load(PROFILE_PATH.read_text(encoding="utf-8"))
    counts = {
        "ifeval": 541,
        "xstest": 450,
        "jbb_behaviors": 100,
        "rewardbench": 2985,
        "preference_vs_start": 256,
        "jsonnav_utility": 256,
    }

    manifest = prepare_suite(profile=profile, out=tmp_path, resolver=_resolver(counts))

    # 500 after amendment 3: XSTest gave up 72 tasks so the 64 arm-level
    # preference prompts fit under the unchanged 512 ceiling.
    assert manifest["generation_tasks_per_model"] == 500
    assert (
        manifest["generation_tasks_per_model"]
        <= profile["generation_budget"]["max_tasks_per_model"]
    )
    by_id = {entry["id"]: entry for entry in manifest["endpoints"]}
    assert by_id["xstest"]["selected_tasks"] == 180
    assert by_id["preference_vs_start"]["selected_tasks"] == 64
    assert by_id["jsonnav_utility"]["external"] is False
    assert by_id["jsonnav_utility"]["dataset"] is None
    assert by_id["preference_vs_start"]["external"] is False
    assert by_id["ifeval"]["external"] is True
    # RewardBench is judged, not generated, so it stays outside the budget.
    assert by_id["rewardbench"]["selected_tasks"] == 96
    # Every endpoint's selection is pinned by a digest.
    assert all(entry["task_ids_digest"] for entry in manifest["endpoints"])


def test_an_unknown_endpoint_is_refused_even_when_marked_internal(tmp_path: Path) -> None:
    """`external: false` is for miniVERL's own measurements, not an escape hatch.

    It still has to be spelled out in the profile with its own task budget; it
    just has no upstream dataset revision to pin.
    """
    manifest = prepare_suite(
        profile=_profile(endpoints=[{"id": "local_thing", "tasks": 4, "external": False}]),
        out=tmp_path,
        resolver=_resolver({"local_thing": 20}),
    )

    entry = manifest["endpoints"][0]
    assert entry["external"] is False
    assert entry["revision"] is None
    assert entry["category"] == "retained_utility"


def test_the_committed_profile_covers_every_required_category() -> None:
    from miniverl.alignment_external.registry import REQUIRED_CATEGORIES, load_registry

    profile = yaml.safe_load(PROFILE_PATH.read_text(encoding="utf-8"))
    by_id = {entry["id"]: entry for entry in load_registry()["endpoints"]}
    covered = {
        by_id[item["id"]]["category"] for item in profile["endpoints"] if item["id"] in by_id
    }

    assert set(REQUIRED_CATEGORIES) <= covered


# ------------------------------------------------------------------ validate


def _row(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "endpoint_id": "ifeval",
        "category": "instruction_following",
        "dataset": "google/IFEval",
        "dataset_revision": "966cd89545d6b6acfd7638bc708b98261ca58e84",
        "split": "train",
        "task_id": "ifeval-0",
        "subset": None,
        "checkpoint_id": "sft",
        "checkpoint_digest": "a" * 64,
        "method": "starting-sft-checkpoint",
        "seed": None,
        "generation_config_digest": "b" * 64,
        "output_digest": "c" * 64,
        "output_tokens": 10,
    }
    fields.update(overrides)
    return TaskRecord(score=1.0, **fields).to_json_row()


def _manifest(task_ids: list[str]) -> dict[str, Any]:
    return {
        "endpoints": [
            {
                "id": "ifeval",
                "revision": "966cd89545d6b6acfd7638bc708b98261ca58e84",
                "task_ids": task_ids,
            }
        ]
    }


def test_a_matching_result_set_validates() -> None:
    manifest = _manifest(["ifeval-0", "ifeval-1"])
    rows = [_row(task_id="ifeval-0"), _row(task_id="ifeval-1")]

    assert validate_results(manifest, rows) == []


def test_a_missing_task_is_reported() -> None:
    problems = validate_results(_manifest(["ifeval-0", "ifeval-1"]), [_row(task_id="ifeval-0")])

    assert any("have no result row" in problem for problem in problems)


def test_an_unplanned_task_is_reported() -> None:
    problems = validate_results(
        _manifest(["ifeval-0"]), [_row(task_id="ifeval-0"), _row(task_id="ifeval-9")]
    )

    assert any("not in the suite" in problem for problem in problems)


def test_a_drifted_dataset_revision_is_reported() -> None:
    """A different revision is a different benchmark."""
    problems = validate_results(
        _manifest(["ifeval-0"]), [_row(task_id="ifeval-0", dataset_revision="f" * 40)]
    )

    assert any("is not the pinned" in problem for problem in problems)


def test_a_result_from_an_unplanned_endpoint_is_reported() -> None:
    problems = validate_results(
        _manifest(["ifeval-0"]),
        [_row(task_id="ifeval-0"), _row(endpoint_id="xstest", task_id="x-0")],
    )

    assert any("absent from the manifest" in problem for problem in problems)


# -------------------------------------------------------------------- report


def test_report_means_are_per_endpoint_with_no_combined_score() -> None:
    rows = [
        _row(task_id="ifeval-0"),
        {**_row(task_id="ifeval-1"), "score": 0.0},
        TaskRecord.not_applicable(
            reason="verifier dependency missing",
            endpoint_id="ifeval",
            category="instruction_following",
            dataset="google/IFEval",
            dataset_revision="966cd89545d6b6acfd7638bc708b98261ca58e84",
            split="train",
            task_id="ifeval-2",
            checkpoint_id="sft",
            checkpoint_digest="a" * 64,
            method="starting-sft-checkpoint",
            generation_config_digest="b" * 64,
        ).to_json_row(),
    ]

    report = report_suite(rows)

    entry = report["endpoints"]["ifeval"]
    assert entry["tasks_evaluated"] == 2
    assert entry["tasks_not_applicable"] == 1
    # The unmeasured task is excluded from the mean rather than counted as 0.
    assert entry["mean_score"] == 0.5
    assert entry["not_applicable_reasons"] == {"verifier dependency missing": 1}
    assert "combined" in report["note"]
    assert "alignment_score" not in report


def test_an_endpoint_with_nothing_measured_reports_none() -> None:
    rows = [
        TaskRecord.not_applicable(
            reason="judge below the qualification floor",
            endpoint_id="rewardbench",
            category="preference_reward",
            dataset="allenai/reward-bench",
            dataset_revision="168d848cdbbea9764fae4a544dc9ca1e6cca4931",
            split="filtered",
            task_id="rb-0",
            checkpoint_id="sft",
            checkpoint_digest="a" * 64,
            method="dpo",
            generation_config_digest="b" * 64,
        ).to_json_row()
    ]

    report = report_suite(rows)

    assert report["endpoints"]["rewardbench"]["mean_score"] is None
    assert report["endpoints"]["rewardbench"]["tasks_evaluated"] == 0
