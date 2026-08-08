"""The pinned sources still have the shape the registry claims.

These run against the real Hugging Face Hub, so they are network tests and are
deselected in CI. They exist because the registry's numbers -- 541 IFEval rows,
450 XSTest prompts split 250 safe / 200 unsafe -- are load-bearing: if upstream
reshapes a split at the pinned revision, every downstream rate silently changes
meaning. Pinning a revision should make that impossible; this proves it.
"""

from __future__ import annotations

import pytest

from miniverl.alignment_external.refusal import is_unsafe_prompt
from miniverl.alignment_external.registry import load_registry

pytestmark = [pytest.mark.network, pytest.mark.slow]

datasets = pytest.importorskip("datasets")


def _endpoint(endpoint_id: str) -> dict:
    for entry in load_registry()["endpoints"]:
        if entry["id"] == endpoint_id:
            return entry
    raise AssertionError(f"no endpoint {endpoint_id} in the registry")


def _load(endpoint_id: str):
    entry = _endpoint(endpoint_id)
    return entry, datasets.load_dataset(
        entry["dataset"],
        entry.get("config"),
        split=entry["split"],
        revision=entry["revision"],
    )


def test_ifeval_row_count_matches_the_registry() -> None:
    entry, ds = _load("ifeval")

    assert ds.num_rows == entry["rows_upstream"] == 541
    assert {"key", "prompt", "instruction_id_list", "kwargs"} <= set(ds.column_names)


def test_every_ifeval_instruction_type_has_a_verifier() -> None:
    """0 of 834 instructions may go unscored, or the endpoint is partial."""
    from miniverl.alignment_external.ifeval import IFEVAL_SUPPORTED_INSTRUCTIONS

    _entry, ds = _load("ifeval")
    present = {item for row in ds for item in row["instruction_id_list"]}

    assert present <= set(IFEVAL_SUPPORTED_INSTRUCTIONS), (
        f"unverifiable instruction types: {sorted(present - set(IFEVAL_SUPPORTED_INSTRUCTIONS))}"
    )


def test_xstest_keeps_its_250_safe_and_200_unsafe_split() -> None:
    entry, ds = _load("xstest")

    assert ds.num_rows == entry["rows_upstream"] == 450
    unsafe = sum(1 for row in ds if is_unsafe_prompt(row["type"]))
    assert (unsafe, ds.num_rows - unsafe) == (200, 250)
    assert len(set(ds["type"])) == entry["strata_count"] == 18


def test_jbb_behaviors_has_its_100_categorised_behaviours() -> None:
    entry, ds = _load("jbb_behaviors")

    assert ds.num_rows == entry["rows_upstream"] == 100
    assert entry["strata_field"] in ds.column_names
    # Categories are what the harmful-compliance profile stratifies on.
    assert len(set(ds[entry["strata_field"]])) >= 5


def test_rewardbench_keeps_its_subsets() -> None:
    entry, ds = _load("rewardbench")

    assert ds.num_rows == entry["rows_upstream"] == 2985
    assert {"prompt", "chosen", "rejected", "subset"} <= set(ds.column_names)
    assert len(set(ds["subset"])) == entry["strata_count"] == 23


def test_the_rejected_candidates_are_still_unavailable() -> None:
    """If a gated source becomes ungated, revisit the substitution.

    This does not fail when access is granted -- it records which of them are
    reachable, so the choice can be revisited deliberately rather than staying
    frozen because nobody looked again.
    """
    registry = load_registry()
    reachable = []
    for candidate in registry["rejected_candidates"]:
        dataset = candidate.get("dataset")
        if not dataset:
            continue
        try:
            datasets.get_dataset_config_names(dataset)
        except Exception:
            continue
        reachable.append(dataset)

    if reachable:
        pytest.skip(
            "these previously gated sources are now reachable and the "
            f"substitution could be revisited: {reachable}"
        )
