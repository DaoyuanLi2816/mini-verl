"""Every upstream leaf is accounted for, even when the full config rejects."""

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from miniverl.bridge.rl_v09 import _classify, _flatten, rl_v09_field_rules_digest

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "benchmarks/compatibility/verl-v0.9.0"


@pytest.mark.parametrize(
    "name",
    ["ppo", "grpo", "dr-grpo", "rloo", "reinforce-plus-plus", "lora-grpo", "trained-rm-grpo"],
)
@pytest.mark.parametrize("scope", ["complete", "launch"])
def test_committed_corpus_accounts_for_every_resolved_field(name, scope):
    source = (CORPUS / f"{name}-{scope}.yaml").read_bytes()
    report = json.loads((CORPUS / f"{name}-{scope}-report.json").read_text())
    fields, unsupported, unresolved = _classify(
        _flatten(yaml.safe_load(source)), profile=report["profile"]
    )
    assert fields == report["field_classification"]
    assert unsupported == report["unsupported_fields"]
    assert unresolved == report["unresolved_fields"]
    assert hashlib.sha256(source).hexdigest() == report["source_config_sha256"]
    assert report["profile_field_rules_sha256"] == rl_v09_field_rules_digest(report["profile"])
    assert not report["executable"]  # Full original experiments are not the bounded adaptations.


def test_shell_reader_never_executes_command_substitutions():
    from scripts.publish_verl_compatibility_corpus import launch_arguments

    with pytest.raises(ValueError, match="non-literal"):
        launch_arguments(
            'X=$(touch /tmp/not-permitted)\nDATA=(\n x=$X\n)\npython3 -m verl.trainer.main_ppo "${DATA[@]}"'
        )
