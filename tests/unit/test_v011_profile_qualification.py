from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("kind", "samples_per_prompt", "rewarded"),
    [
        ("direct_n1", 1, False),
        ("grouped_pg_n4", 4, False),
        ("rewarded_pg_n4", 4, True),
    ],
)
def test_v011_profile_qualification_compiles_exact_intended_profile(
    tmp_path: Path,
    kind: str,
    samples_per_prompt: int,
    rewarded: bool,
) -> None:
    from scripts.run_v011_profile_qualification import _build_plan, _write_dataset

    dataset = tmp_path / kind / "data.parquet"
    dataset.parent.mkdir()
    _write_dataset(dataset, rewarded=rewarded)
    plan, config = _build_plan(kind, dataset, tmp_path / kind / "plan.json")

    assert config.rollout.backend == "hf_cached"
    assert config.rollout.samples_per_prompt == samples_per_prompt
    assert config.source.use_task_rewards is rewarded
    assert config.train.cycles == 1
    assert plan.resolved_native_config == config.model_dump(mode="json")
