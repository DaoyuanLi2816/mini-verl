from __future__ import annotations

from pathlib import Path

import yaml


def test_v012_gpu_workload_source_compiles_to_validated_grpo_recipe(tmp_path: Path) -> None:
    from miniverl.algorithms.contract import UPSTREAM_VERL_COMMIT
    from miniverl.bridge.rl_v09 import publish_imported_verl_rl_v09
    from miniverl.config import RunConfig
    from scripts.run_v012_rl_qualification import _source_payload

    source = tmp_path / "source.yaml"
    source.write_text(
        yaml.safe_dump(_source_payload(tmp_path / "data.parquet"), sort_keys=False),
        encoding="utf-8",
    )
    recipe = tmp_path / "native.yaml"

    report = publish_imported_verl_rl_v09(
        source,
        out=recipe,
        target_verl=UPSTREAM_VERL_COMMIT,
    )
    config = RunConfig.from_yaml(recipe)

    assert report["status"] == "accepted"
    assert report["generated_recipe_validated"] is True
    assert config.algorithm.name.value == "grpo"
    assert config.reward.provider.value == "target_length"
    assert config.rollout.samples_per_prompt == 4
    assert config.train.cycles == 2
    assert config.train.gradient_accumulation_steps == 8
