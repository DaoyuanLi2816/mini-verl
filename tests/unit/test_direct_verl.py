"""No-rewrite import separates semantic rejection from deployment inputs."""

from pathlib import Path

import pytest
import yaml

from miniverl.bridge.direct import compile_direct
from miniverl.errors import ConfigError

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "case",
    ["ppo", "grpo", "dr-grpo", "rloo", "reinforce-plus-plus", "lora-grpo", "trained-rm-grpo"],
)
def test_complete_real_config_acceptance_is_independent_of_missing_data(case):
    source = ROOT / "benchmarks/compatibility/verl-v0.9.0" / f"{case}-complete.yaml"
    result = compile_direct(source)
    assert result["semantic_status"] == "accepted", result.get("rejections")
    assert result["execution_status"] == "requires_bindings"
    assert result["required_bindings"]
    assert result["source_config_sha256"]
    assert len(result["fields"]) == result["source_leaf_count"]


def test_unknown_semantics_and_enabled_group_filter_reject(tmp_path):
    source = ROOT / "benchmarks/compatibility/verl-v0.9.0/grpo-complete.yaml"
    value = yaml.safe_load(source.read_text())
    value["algorithm"]["unknown_loss"] = 7
    path = tmp_path / "unknown.yaml"
    path.write_text(yaml.safe_dump(value))
    result = compile_direct(path)
    assert result["semantic_status"] == "rejected"
    assert any(r["field"] == "algorithm.unknown_loss" for r in result["rejections"])


def test_duplicate_yaml_and_nonfinite_binding_fail(tmp_path):
    path = tmp_path / "duplicate.yaml"
    path.write_text("algorithm: {adv_estimator: grpo}\nalgorithm: {adv_estimator: gae}\n")
    with pytest.raises(ConfigError, match="duplicate"):
        compile_direct(path)


@pytest.mark.parametrize(
    "text",
    [
        "algorithm: &a {adv_estimator: grpo}\nx: *a",
        "algorithm: {adv_estimator: grpo}\ndata: {train_files: '${dataset}'}",
    ],
)
def test_unresolved_or_aliased_yaml_has_an_actionable_error(tmp_path, text):
    path = tmp_path / "unresolved.yaml"
    path.write_text(text)
    with pytest.raises(ConfigError, match="resolve"):
        compile_direct(path)


def test_legacy_local_flags_cannot_be_silently_reinterpreted(tmp_path):
    value = yaml.safe_load((ROOT / "src/miniverl/resources/verl_direct_grpo.yaml").read_text())
    value["miniverl"] = {"actor": {"lora_enabled": False}, "memory": {"strategy": "resident"}}
    path = tmp_path / "legacy.yaml"
    path.write_text(yaml.safe_dump(value))
    report = compile_direct(path)
    assert report["semantic_status"] == "rejected"
    assert {row["field"] for row in report["rejections"]} >= {
        "miniverl.actor.lora_enabled",
        "miniverl.memory.strategy",
    }
