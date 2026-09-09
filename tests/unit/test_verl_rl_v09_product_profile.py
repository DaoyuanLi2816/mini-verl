"""The new profile follows upstream prompt-based minibatch units."""

from pathlib import Path

import pytest
import yaml

from miniverl.bridge.rl_v09 import publish_imported_verl_rl_v09
from miniverl.utils.runs import write_text

PROFILE = "verl-rl-v0.9-single-gpu-v3"
ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "field,value", [("loss_agg_mode", "seq-mean-token-mean"), ("ppo_epochs", 2)]
)
def test_v3_rejects_unimplemented_grpo_aggregation_or_repeated_epochs(tmp_path, field, value):
    payload = yaml.safe_load((ROOT / "src/miniverl/resources/verl_grpo.yaml").read_text())
    payload["actor_rollout_ref"]["actor"][field] = value
    source = tmp_path / "source.yaml"
    out = tmp_path / "local.yaml"
    write_text(source, yaml.safe_dump(payload))
    result = publish_imported_verl_rl_v09(source, out=out, target_verl="v0.9.0", profile=PROFILE)
    assert result["status"] == "rejected"
    assert not out.exists()
    rows = {row["upstream_field"]: row for row in result["field_classification"]}
    assert rows[f"actor_rollout_ref.actor.{field}"]["classification"] == "not_implemented"


def test_v3_ppo_expands_prompt_minibatches_and_reports_gamma(tmp_path):
    payload = yaml.safe_load((ROOT / "examples/verl-rl-v0.9-single-gpu-ppo.yaml").read_text())
    payload["actor_rollout_ref"]["actor"]["ppo_mini_batch_size"] = 2
    payload["critic"]["ppo_mini_batch_size"] = 2
    payload["actor_rollout_ref"]["actor"]["fsdp_config"] = {"param_offload": True}
    source = tmp_path / "source.yaml"
    out = tmp_path / "local.yaml"
    write_text(source, yaml.safe_dump(payload))
    result = publish_imported_verl_rl_v09(source, out=out, target_verl="v0.9.0", profile=PROFILE)
    assert result["status"] == "accepted"
    local = yaml.safe_load(out.read_text())
    assert local["train"]["gradient_accumulation_steps"] == 4
    rows = {row["upstream_field"]: row for row in result["field_classification"]}
    assert rows["algorithm.gamma"]["classification"] == "exact"
    assert (
        rows["actor_rollout_ref.actor.fsdp_config.param_offload"]["classification"]
        == "locally_lowered"
    )
