"""Two endpoints must not silently share prompts.

RewardBench's filtered split contains `xstest-should-refuse` and
`xstest-should-respond` -- 404 of its 2,985 rows are the same prompts as this
study's over-refusal endpoint. Left in, one behaviour change would move both
the over-refusal rate and the preference win rate, and a reader would take two
independent-looking endpoints as corroboration when they are the same data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from miniverl.alignment_external.registry import load_registry
from miniverl.alignment_external.suite import prepare_suite


def _rewardbench() -> dict[str, Any]:
    for entry in load_registry()["endpoints"]:
        if entry["id"] == "rewardbench":
            return entry
    raise AssertionError("rewardbench is missing from the registry")


def test_the_registry_excludes_the_overlapping_subsets() -> None:
    excluded = set(_rewardbench()["exclude_strata"])

    assert excluded == {"xstest-should-refuse", "xstest-should-respond"}
    assert "over-refusal" in _rewardbench()["exclude_reason"]


def _resolver(strata_by_endpoint: dict[str, list[str]]) -> Any:
    def resolve(endpoint: dict[str, Any]) -> tuple[list[str], list[str] | None]:
        strata = strata_by_endpoint[endpoint["id"]]
        ids = [f"{endpoint['id']}-{index:04d}" for index in range(len(strata))]
        return ids, strata

    return resolve


def test_excluded_strata_never_reach_the_manifest(tmp_path: Path) -> None:
    strata = ["chat"] * 40 + ["xstest-should-refuse"] * 30 + ["xstest-should-respond"] * 30
    profile = {
        "id": "overlap-test",
        "selection_seed": 3,
        "endpoints": [{"id": "rewardbench", "tasks": 20}],
    }

    manifest = prepare_suite(
        profile=profile, out=tmp_path, resolver=_resolver({"rewardbench": strata})
    )

    entry = manifest["endpoints"][0]
    assert entry["excluded_tasks"] == 60
    assert entry["excluded_strata"] == ["xstest-should-refuse", "xstest-should-respond"]
    # Only the 40 `chat` rows were eligible, and they are ids 0000-0039.
    assert all(int(task_id.split("-")[-1]) < 40 for task_id in entry["task_ids"])


def test_excluding_everything_fails_closed(tmp_path: Path) -> None:
    strata = ["xstest-should-refuse"] * 20
    profile = {
        "id": "overlap-test",
        "selection_seed": 3,
        "endpoints": [{"id": "rewardbench", "tasks": 5}],
    }

    with pytest.raises(ValueError, match="removed every task"):
        prepare_suite(profile=profile, out=tmp_path, resolver=_resolver({"rewardbench": strata}))


def test_exclusion_without_strata_is_refused(tmp_path: Path) -> None:
    """Filtering by stratum needs the stratum labels to filter on."""

    def resolve(endpoint: dict[str, Any]) -> tuple[list[str], list[str] | None]:
        return [f"r-{index}" for index in range(20)], None

    profile = {
        "id": "overlap-test",
        "selection_seed": 3,
        "endpoints": [{"id": "rewardbench", "tasks": 5}],
    }

    with pytest.raises(ValueError, match="needs a strata_field"):
        prepare_suite(profile=profile, out=tmp_path, resolver=resolve)


@pytest.mark.network
@pytest.mark.slow
def test_the_overlap_is_real_upstream() -> None:
    """The exclusion exists because of actual upstream content, not a guess."""
    datasets = pytest.importorskip("datasets")

    entry = _rewardbench()
    dataset = datasets.load_dataset(
        entry["dataset"], entry.get("config"), split=entry["split"], revision=entry["revision"]
    )
    subsets = list(dataset["subset"])
    overlapping = sum(1 for name in subsets if name in set(entry["exclude_strata"]))

    assert overlapping == 404, f"expected 404 XSTest-derived rows, found {overlapping}"
    assert len(subsets) - overlapping == 2581


@pytest.mark.network
@pytest.mark.slow
def test_the_committed_profile_excludes_the_overlap_end_to_end(tmp_path: Path) -> None:
    """Prepared against the real Hub, no XSTest prompt reaches the preference endpoint."""
    datasets = pytest.importorskip("datasets")

    profile = yaml.safe_load(
        Path("benchmarks/external-alignment/profile-v1.yaml").read_text(encoding="utf-8")
    )

    def resolve(endpoint: dict[str, Any]) -> tuple[list[str], list[str] | None]:
        if endpoint.get("dataset") is None:
            return [f"{endpoint['id']}-{index:04d}" for index in range(256)], None
        loaded = datasets.load_dataset(
            endpoint["dataset"],
            endpoint.get("config"),
            split=endpoint["split"],
            revision=endpoint["revision"],
        )
        ids = [f"{endpoint['id']}-{index:05d}" for index in range(loaded.num_rows)]
        field = endpoint.get("strata_field")
        return ids, ([str(value) for value in loaded[field]] if field else None)

    manifest = prepare_suite(profile=profile, out=tmp_path, resolver=resolve)

    rewardbench = next(e for e in manifest["endpoints"] if e["id"] == "rewardbench")
    assert rewardbench["excluded_tasks"] == 404
    assert rewardbench["selected_tasks"] == 96
